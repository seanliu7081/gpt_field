"""Causal policy regressions; no model, simulator, or network is invoked."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from gpt_field.agent_speedup_v3 import AgentSpeedUpV3Policy, _action_identity, _blocker_identity
from gpt_field.agent_speedup_v3_geometry import GeometryMemory
from gpt_field.agent_speedup_v3_observation import propose_evidence_views
from gpt_field.contact_skills import StepView


class Watch:
    def capture(self, *args, **kwargs):
        return {"status": "reference", "geometry_verified": False}

    def assess(self, *args, **kwargs):
        return {"status": "consistent", "geometry_verified": False}

    def forget(self, *args):
        pass


def entity(ident, center, half=(.5, .5, .5)):
    return {"id": ident, "role": "free_region", "parent_id": None, "frame": "world",
            "position_m": list(center), "quaternion_wxyz": [1, 0, 0, 0],
            "boxes": [{"center_local_m": [0, 0, 0], "half_extents_m": list(half), "quaternion_wxyz": [1, 0, 0, 0]}],
            "uncertainty_m": .001, "coverage": "visible_surfaces", "attachment": "none",
            "related_ids": [], "anchors": None}


def setup_policy():
    policy = AgentSpeedUpV3Policy.__new__(AgentSpeedUpV3Policy)
    view = StepView(SimpleNamespace(state=[0., 0., .8, 0., 0., 0., .04, -.04]))
    context = {"intent": "transit", "target_id": None, "held_object_id": None, "allowed_contact_ids": [],
               "scene_inventory": {"reviewed": False, "support_ids": [], "nearby_obstacle_ids": [], "unresolved": True}}
    policy.shared_geometry = GeometryMemory(visual_watch=Watch())
    policy.operation_runtime = SimpleNamespace(snapshot=lambda: {"task_success": "not_available_to_policy"})
    policy.step_index = 4
    policy._intent_action = {"type": "motion", "motion": {"xyz_m": [.15, 0., .8]}, "execution": None, "macro": None}
    policy._pending_intent = {"intent_id": "pending", "kind": "motion", "motion_context": context,
        "saved": {"current_target": {"xyz_m": [.15, 0., .8], "quaternion_wxyz": [1, 0, 0, 0]}}}
    policy._primitive = None
    policy._last_motion_tracking = {}
    policy._evidence_pause = None
    policy.motion_context = context
    policy._record_feedback = lambda value: setattr(policy, "feedback", value)
    policy.shared_geometry.assess_motion(view, [.15, 0., .8], [1, 0, 0, 0], context, 4)
    key = _action_identity(policy._intent_action, context)
    policy._repeat_guard = {"action": key, "blocker": _blocker_identity(policy.shared_geometry.last_motion),
        "evidence": policy._relevant_evidence(view, context), "rejections": 0}
    return policy, view, context, key


def test_new_free_volume_is_evaluated_before_repeat_rejection_without_mutating_live_memory():
    policy, view, context, key = setup_policy()
    patch = {"entities": [entity("new_air", [0, 0, .8])], "invalidate_ids": [], "remove_ids": []}
    old = deepcopy(policy.shared_geometry.snapshot())
    candidate = policy.shared_geometry.preview_scene_update(patch, view, 4)
    assert not policy._reject_repeated(key, view, context, patch, candidate)
    assert policy.shared_geometry.snapshot() == old
    assert "new_air" in candidate.entities and "new_air" not in policy.shared_geometry.entities


def test_remote_new_region_does_not_reset_blocked_action_guard():
    policy, view, context, key = setup_policy()
    patch = {"entities": [entity("unrelated", [8, 8, 8])], "invalidate_ids": [], "remove_ids": []}
    candidate = policy.shared_geometry.preview_scene_update(patch, view, 4)
    assert policy._reject_repeated(key, view, context, patch, candidate)
    assert policy.feedback["reason"] == "unchanged_blocker_requires_new_evidence_or_intent"


def test_duplicate_renamed_region_has_no_new_path_effect():
    policy, view, context, _ = setup_policy()
    patch = {"entities": [entity("existing", [0, 0, .8])], "invalidate_ids": [], "remove_ids": []}
    policy.shared_geometry.update_from_model(patch, view, 4)
    alias = {"entities": [entity("alias", [0, 0, .8])], "invalidate_ids": [], "remove_ids": []}
    assert not policy._patch_changes_path(policy.shared_geometry.preview_scene_update(alias, view, 4), view, context)


def test_whole_goal_evidence_is_retained_instead_of_supported_local_prefix():
    policy, _, _, _ = setup_policy()
    goal = {"status": "unknown", "reason": "uncovered", "coverage_witnesses": [{"body_part": "hand_collision", "axis": "y"}],
            "missing_evidence_requests": [{"entity_id": "air", "kind": "coverage", "body_part": "hand_collision"}]}
    policy._pending_intent["saved"]["whole_goal_assessment"] = goal
    policy.shared_geometry.last_motion = {"status": "supported", "execution_prefix": {"status": "supported", "fraction": .4}}
    core = policy._decision_core()
    assert core["whole_goal_assessment"] == goal
    assert core["missing_evidence_requests"] == goal["missing_evidence_requests"]
    assert core["executable_prefix_assessment"]["status"] == "supported"


def test_missing_semantic_geometry_preserves_program_without_fabricating_pose():
    policy, view, context, _ = setup_policy()
    saved = policy._pending_intent["saved"]
    missing = ["geometry_entity:object"]
    saved.update(current_target=None, program={"stages": [{"goal": {"kind": "grasp", "reference_id": "object"}}]},
                 stage_index=0, phase="move", completed_stages=[], missing_evidence_requests=missing,
                 whole_goal_assessment={"status": "unknown", "reason": "program_geometry_required",
                                        "missing_evidence_requests": missing})
    policy._pending_intent["kind"] = "program"
    result = policy._decision_core()
    assert result["pending_intent"]["program"] == saved["program"]
    assert result["missing_evidence_requests"] == missing
    assert result["pending_intent"]["current_target"] is None
    assert policy._propose_views(view)["candidates"] == []
    assert policy._propose_views(view)["missing_evidence_requests"] == missing
    assert not policy._patch_changes_path(policy.shared_geometry, view, context)


def test_view_request_retains_measured_failure_despite_admissible_nominal_preview(monkeypatch):
    import gpt_field.agent_speedup_v3 as module
    policy, view, _, _ = setup_policy()
    policy.ctx = SimpleNamespace(memory={"user_check_gripper_command": "open"})
    policy._view_proposal_key = None
    policy._evidence_view_attempts = {}
    policy._last_motion_tracking = {"status": "mismatch", "sample_count": 12,
                                    "commanded_translation_m": .0048, "measured_translation_m": .000006}
    observed = []
    def propose(geometry, snapshot, context, assessment, step, attempted):
        observed.append(assessment)
        assert "tracking" not in context  # A new view is a new measured motion.
        return {"candidates": [], "reason": "fixture"}
    monkeypatch.setattr(module, "propose_evidence_views", propose)
    policy._propose_views(view)
    assert observed[0]["tracking_status"] == "mismatch"
    assert "measured_tracking_mismatch" in observed[0]["blocking_reasons"]
    assert observed[0]["failed_goal_xyz_m"] == [.15, 0., .8]


def test_local_framework_snapshot_skips_full_history_but_rebuilds_for_model(monkeypatch):
    import gpt_field.agent_speedup_v3 as module
    policy, view, _, _ = setup_policy()
    policy.plan_lock = SimpleNamespace(_world_state={"reviewed": "preserved"}, _last_feedback=None)
    policy.ctx = SimpleNamespace(memory={"user_check_gripper_command": "open"})
    policy._feedback = [{"status": "moving"}]
    policy._need_agent = False
    policy._framework_sync_s = 0.
    policy._framework_full_syncs = policy._framework_light_syncs = 0
    calls = []
    def full(current, latest):
        calls.append(latest)
        current.plan_lock._world_state = {"native_estimate": {"shared_geometry": {"revision": 99}}}
    monkeypatch.setattr(module.AgentFirstV2Policy, "_sync_framework", full)
    policy._sync_framework(view)
    assert calls == [] and policy._framework_light_syncs == 1
    assert policy.plan_lock._world_state["reviewed"] == "preserved"
    assert policy.plan_lock._world_state["measured_robot"]["observed_at_step"] == policy.step_index
    policy._need_agent = True
    policy._sync_framework(view)
    assert calls == [view] and policy._framework_full_syncs == 1
    assert policy.plan_lock._world_state["shared_geometry"] == {"revision": 99}
    assert policy.plan_lock._world_state["local_snapshot"] is False


def test_shared_operation_monitor_receives_nonprogram_action_and_validated_transition(monkeypatch):
    import gpt_field.agent_speedup_v3 as module
    policy, view, context, _ = setup_policy()
    policy.ctx = SimpleNamespace(memory={"user_check_gripper_command": "open"})
    policy._view = lambda observation: observation
    events = []
    policy.operation_runtime = SimpleNamespace(
        before_action=lambda before, ctx, command, step: events.append(("before", command, step, ctx)),
        observe_transition=lambda before, after, ctx, command, step: events.append(("observed", command, step)))
    def act(current, observation):
        current.step_index += 1
        return SimpleNamespace(action=(0., 0., 0., 0., 0., 0., 1.))
    def acknowledge(current, before, decision, after):
        events.append(("base_acknowledged",))
        current.ctx.memory["user_check_gripper_command"] = "close"
        return {}
    monkeypatch.setattr(module.AgentFirstV2Policy, "act", act)
    monkeypatch.setattr(module.AgentFirstV2Policy, "observe_transition", acknowledge)
    policy.shared_geometry.observe_transition = lambda *args: events.append(("geometry_updated",))
    policy._sync_framework = lambda after: events.append(("snapshot",))
    decision = policy.act(view)
    assert events == [("before", "close", 4, context)]
    result = policy.observe_transition(view, decision, view)
    assert events[1:] == [("base_acknowledged",), ("geometry_updated",), ("observed", "close", 5), ("snapshot",)]
    assert result["policy_version"] == module.POLICY_VERSION


@pytest.mark.parametrize("accepted", [True, False])
def test_only_accepted_local_evidence_resumes_unchanged_supported_goal(monkeypatch, accepted):
    import gpt_field.agent_speedup_v3 as module
    policy, view, context, _ = setup_policy()
    context["target_id"] = "object"
    policy.shared_geometry.entities["object"] = {"local_pose_revision": 2,
        "observation_step": policy.step_index, "local_pose_evidence": {
            "accepted": accepted, "observed_at_step": policy.step_index}}
    policy.shared_geometry.preview_motion = lambda *args: {"status": "supported"}
    policy._expected_observation = None
    policy.ctx = SimpleNamespace(view=view, memory={"user_check_gripper_command": "open"})
    policy._active_intent_id = "same-operation-intent"
    policy._local_evidence_resumes = set()
    policy._continuation_events = []
    saved = {"current_target": {"xyz_m": [.15, 0., .8], "quaternion_wxyz": [1, 0, 0, 0]},
             "remaining_watchdog_steps": 17}
    calls = []
    class Resumed:
        done = False
        @classmethod
        def from_continuation(cls, current, state, geometry, current_context, **options):
            calls.append(state)
            assert state == saved and state is not saved
            assert current_context is context
            assert options["operation_runtime"] is policy.operation_runtime
            return cls()
    monkeypatch.setattr(module, "AdaptiveLocalMotionPrimitive", Resumed)
    stopped = SimpleNamespace(outcome=lambda: {"reason": "geometry_observation_horizon"})
    assert policy._resume_with_local_evidence(stopped, saved, "motion") is accepted
    assert len(calls) == int(accepted)
    if accepted:
        assert policy._need_agent is False
        assert policy.feedback["remaining_watchdog_steps"] == 17
        assert not policy._resume_with_local_evidence(stopped, saved, "motion")


@pytest.mark.parametrize("boundary_uncertainty", [.01, None])
def test_observation_requires_camera_and_reachable_supported_path(boundary_uncertainty):
    policy, view, context, _ = setup_policy()
    gap = {"coverage_witnesses": [{"world_point_m": [0, 0, .3], "body_part": "hand_collision", "uncertainty_m": boundary_uncertainty}]}
    assert propose_evidence_views(policy.shared_geometry, view, context, gap, 4)["reason"] == "synchronized_wrist_camera_unavailable"
    view.observation.geometry = {"image": {"width": 512, "height": 512}, "cameras": {"wrist": {
        "position_world_m": [0, 0, .8], "rotation_world_from_camera": np.eye(3).tolist(),
        "focal_length_px": [300, 300], "principal_point_px": [256, 256]}}}
    assert not propose_evidence_views(policy.shared_geometry, view, context, gap, 4)["candidates"]
    patch = {"entities": [entity("room", [0, 0, .7], (1., 1., 1.))], "invalidate_ids": [], "remove_ids": []}
    policy.shared_geometry.update_from_model(patch, view, 4)
    proposals = propose_evidence_views(policy.shared_geometry, view, context, gap, 4)
    assert proposals["candidates"]
    assert proposals["boundary_uncertainty_available"] == (boundary_uncertainty is not None)
    assert gap["coverage_witnesses"][0]["uncertainty_m"] == boundary_uncertainty
    for item in proposals["candidates"]:
        assert item["path_status"] == "supported" and item["expected_pixel_shift"] >= 8
        assert item["execution"]["segments"][0]["gripper"] == "keep"
    assert proposals["original_intent_is_not_resumed"]


def test_evidence_view_preserves_goal_contacts_and_watchdog_without_autoplay(monkeypatch):
    import gpt_field.agent_speedup_v3 as module
    policy, view, context, _ = setup_policy()
    context.update(target_id="object", allowed_contact_ids=["object"])
    policy._pending_intent["saved"].update(remaining_watchdog_steps=17, consumed_steps=8)
    original = deepcopy(policy._pending_intent)
    policy._evidence_view_attempts = {}
    policy._evidence_view_session = None
    policy._fragment_grants = 0
    policy.ctx = SimpleNamespace(memory={"user_check_gripper_command": "open"})
    selected = {"xyz_m": [.03, 0, .8], "execution": {"segments": []}, "path_status": "supported"}
    policy._propose_views = lambda current: {"candidates": [selected], "reason": "supported_view_candidates"}
    class ViewingPrimitive:
        done = False
        def __init__(self, *args, **kwargs):
            assert args[3]["allowed_contact_ids"] == []
        def outcome(self):
            return {"status": "succeeded", "reason": "observation_required", "actions": 9}
    monkeypatch.setattr(module, "AdaptiveFragmentSequencePrimitive", ViewingPrimitive)
    def base_complete(current):
        current._primitive = None
        current._need_agent = True
    monkeypatch.setattr(module.AgentFirstV2Policy, "_complete_direct", base_complete)
    policy._acquire_evidence({"continuation": {"intent_id": "pending", "resume": False},
        "motion_context": context, "scene_update": {"entities": [], "invalidate_ids": [], "remove_ids": []}}, view)
    assert policy._pending_intent is None
    assert policy.motion_context["intent"] == "observe"
    policy._complete_direct()
    assert policy._pending_intent == original
    assert policy.motion_context["allowed_contact_ids"] == ["object"]
    assert policy._primitive is None and policy._need_agent
    assert policy.feedback["original_goal_resumed"] is False


def test_unverified_prior_release_stops_next_object_close_before_decision_returns(monkeypatch):
    import gpt_field.agent_speedup_v3 as module
    from gpt_field.agent_speedup_v3_operation import OperationRuntime
    from test_agent_speedup_v3_operation import Geometry, Monitor, settle

    policy, view, context, _ = setup_policy()
    geometry = Geometry()
    runtime = OperationRuntime(geometry, monitor=Monitor(geometry))
    first = {**context, "intent": "grasp", "target_id": "package"}
    settle(runtime, view, first)
    runtime.before_action(view, first, "open", 12)
    runtime.observe_transition(view, view, first, "open", 13)
    policy.operation_runtime = runtime
    policy.motion_context = {**first, "target_id": "other"}
    policy.step_index = 13
    policy.ctx = SimpleNamespace(memory={"user_check_gripper_command": "open"})
    policy._view = lambda observation: observation
    old_operation = runtime.operation_id
    def proposes_close(current, observation):
        current.step_index += 1
        return SimpleNamespace(action=(0., 0., 0., 0., 0., 0., 1.))
    monkeypatch.setattr(module.AgentFirstV2Policy, "act", proposes_close)
    with pytest.raises(module.EvidenceAcquisitionFailed, match="active_manipulated_object_binding_conflict"):
        policy.act(view)
    assert policy._evidence_pause["automatic_retry"] is False
    assert runtime.bound_object_id == "package" and runtime.operation_id == old_operation
    assert runtime.last_command == "open"
    assert runtime.counts["references_started"] == 2
    assert runtime.counts["release_supported"] == 0


def test_compact_pending_contract_preserves_blocker_budget_and_full_runtime_state():
    policy, _, _, _ = setup_policy()
    contract = {"operations": [{"id": "first", "object_id": "a"}, {"id": "second", "object_id": "b"}]}
    saved = policy._pending_intent["saved"]
    saved.update(operation_contract=contract, program={"stages": [{"details": i} for i in range(16)]},
                 stage_index=9, stage_operation_ids=["first"]*8+["second"]*8,
                 completed_operations=[{"id": "first", "completed_at_step": 12}],
                 completed_stages=[{}]*9, remaining_stages=[{}]*7,
                 remaining_watchdog_steps=17, consumed_steps=33)
    policy._pending_intent["kind"] = "program"
    policy._intent_action = {"type": "program", "program": contract}
    before = deepcopy(policy._pending_intent)
    result = policy._decision_core()
    pending = result["pending_intent"]
    assert pending["program"] == contract
    assert pending["current_operation_id"] == "second" and pending["remaining_operation_ids"] == ["second"]
    assert pending["remaining_watchdog_steps"] == 17 and pending["consumed_steps"] == 33
    assert "remaining_stages" not in pending and pending["remaining_stage_count"] == 7
    assert result["missing_evidence_requests"] == policy._goal_assessment()["missing_evidence_requests"]
    assert result["requirements_truncated"] is False
    assert policy._pending_intent == before
    assert result["current_requested_action"]["program"] == {"contract_source": "pending_intent.program"}
