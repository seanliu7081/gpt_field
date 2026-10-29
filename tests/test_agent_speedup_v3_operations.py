"""Focused compact-program contracts; no simulator, model, or task fixtures."""
from copy import deepcopy

import pytest

import gpt_field.agent_speedup_v3_program as program_module
from gpt_field.agent_speedup_v3_program import (
    AdaptiveProgramPrimitive, compile_operations, program_schema, validate_program,
)
from test_agent_speedup_v3_program import Geometry, Monitor, advance, view


def operation(ident="first", obj="object", destination="container", after=()):
    return {"id": ident, "kind": "pick_place", "object_id": obj, "destination_id": destination,
        "after": list(after), "quaternion_wxyz": None,
        "source_inventory": {"reviewed": True, "support_ids": ["table"],
            "nearby_obstacle_ids": ["source_neighbor"], "unresolved": False},
        "destination_inventory": {"reviewed": True, "support_ids": ["shelf"],
            "nearby_obstacle_ids": ["wall"], "unresolved": False},
        "route_obstacle_ids": ["source_neighbor", "wall"],
        "contact_ids": {"grasp": [obj], "carry": [obj], "place": [obj, "shelf"], "retreat": []},
        "max_steps": 40}


def measured_goal(monkeypatch):
    # These tests isolate execution ordering and accepted-evidence gates. Actual
    # OBB goal derivation and swept geometry remain covered by their own tests.
    monkeypatch.setattr(program_module, "derive_program_goal", lambda measured, *args, **kwargs: {
        "xyz_m": measured.position.tolist(), "quaternion_wxyz": [0., 1., 0., 0.],
        "evidence_ids": [], "pose_evidence_revisions": {}, "derivation": "test_measured_pose"})


def drive_until_pause(primitive, limit=100):
    measured = view(gap=.04)
    for _ in range(limit):
        if primitive.done:
            return
        advance(primitive, measured, measured)
    raise AssertionError("finite compact program did not terminate")


def test_compiler_preserves_authorized_objects_contacts_and_verification_boundaries():
    compact = {"operations": [operation()]}
    compiled = compile_operations(compact)
    stages = compiled["program"]["stages"]
    assert validate_program(compact) == compact
    assert program_schema()["anyOf"][1]["properties"]["operations"]["maxItems"] == 3
    assert [s["goal"]["kind"] for s in stages] == ["approach", "grasp", "hold", "lift_clear", "transfer", "place", "hold", "retreat"]
    assert [s["verification"] for s in stages] == ["arrival", "arrival", "arrival", "visual_hold", "arrival", "arrival", "arrival", "visual_release"]
    assert [s["motion_context"]["held_object_id"] for s in stages] == [None, None, None]+["object"]*5
    assert stages[1]["motion_context"]["allowed_contact_ids"] == ["object"]
    assert stages[5]["motion_context"]["allowed_contact_ids"] == ["object", "shelf"]
    assert stages[7]["motion_context"]["allowed_contact_ids"] == []
    assert stages[4]["motion_context"]["scene_inventory"] == {
        "reviewed": True, "support_ids": ["table", "shelf"],
        "nearby_obstacle_ids": ["source_neighbor", "wall"], "unresolved": False}
    assert compiled["stage_operation_ids"] == ["first"]*8
    assert all(set(s["goal"]) == {"kind", "reference_id", "quaternion_wxyz"} for s in stages)
    assert compact == {"operations": [operation()]}


def test_operation_dependencies_sort_stably_and_reject_missing_or_cyclic_edges():
    first, second = operation(), operation("second", "other", after=["first"])
    compiled = compile_operations({"operations": [second, first]})
    assert compiled["stage_operation_ids"] == ["first"]*8+["second"]*8
    for ops in ([first, first], [operation(after=["absent"])], [operation(after=["first"])],
                [operation(after=["second"]), second]):
        with pytest.raises(ValueError, match="dependencies|unique"):
            validate_program({"operations": ops})


@pytest.mark.parametrize("mutate", [
    lambda item: item.update(xyz_m=[0., 0., 0.]),
    lambda item: item.update(task_id="task_1"),
    lambda item: item.update(kind="open_drawer"),
    lambda item: item.update(max_steps=True),
    lambda item: item["contact_ids"].update(grasp=["object", "object"]),
    lambda item: item.update(quaternion_wxyz=[0., 0., 0., 0.]),
])
def test_compact_contract_rejects_unapproved_fields_and_invalid_bounds(mutate):
    item = operation()
    mutate(item)
    with pytest.raises(ValueError):
        validate_program({"operations": [item]})


class ReleaseGateMonitor(Monitor):
    def __init__(self):
        super().__init__("supported")
        self.release_status = "unknown"

    def evidence(self, kind, *args):
        evidence = super().evidence(kind, *args)
        if kind == "release":
            evidence["status"] = self.release_status
        return evidence


def test_second_operation_waits_for_accepted_release_and_resumes_without_model_plan(monkeypatch):
    measured_goal(monkeypatch)
    geometry, monitor = Geometry(), ReleaseGateMonitor()
    compact = {"operations": [operation(), operation("second", "other", after=["first"])]}
    primitive = AdaptiveProgramPrimitive(view(gap=.04), compact, geometry, interaction_monitor=monitor)
    drive_until_pause(primitive)
    assert primitive.reason == "program_visual_verification_required"
    assert primitive.stage_index == 7 and not primitive.completed_operations
    assert primitive.motion_context["held_object_id"] == "object"
    assert primitive.progress()["status"] == "pending_verification"
    assert primitive.progress()["primitive_status"] == "succeeded"
    assert not any(ident == "other" for ident, _, _ in monitor.begins)
    saved = primitive.export_continuation()
    assert saved["operation_contract"] == compact
    assert saved["stage_operation_ids"] == ["first"]*8+["second"]*8
    assert saved["consumed_steps"] == primitive.actions_executed
    monitor.release_status = "supported"
    resumed = AdaptiveProgramPrimitive.from_continuation(view(gap=.04), saved, geometry,
        step=primitive.actions_executed, interaction_monitor=monitor)
    assert resumed.stage_index == 8 and [item["id"] for item in resumed.completed_operations] == ["first"]
    assert resumed.payload_release_events[0]["object_id"] == "object"
    assert resumed.program["stages"][7]["motion_context"]["held_object_id"] is None
    drive_until_pause(resumed)
    assert resumed.reason == "program_complete" and resumed.progress()["status"] == "succeeded"
    assert [item["id"] for item in resumed.completed_operations] == ["first", "second"]
    assert [item["release_evidence"]["target_id"] for item in resumed.completed_operations] == ["object", "other"]
    assert resumed._prior_actions == saved["consumed_steps"]


def test_visual_release_rejected_by_geometry_cannot_start_next_operation(monkeypatch):
    measured_goal(monkeypatch)
    geometry, monitor = Geometry(), ReleaseGateMonitor()
    monitor.release_status = "supported"
    geometry.confirm_attachment_evidence = lambda evidence, *args: {
        "accepted": evidence["kind"] != "release", "target_id": evidence["target_id"]}
    primitive = AdaptiveProgramPrimitive(view(gap=.04), {"operations": [operation(), operation("second", "other")]},
        geometry, interaction_monitor=monitor)
    drive_until_pause(primitive)
    assert primitive.stage_index == 7 and not primitive.completed_operations
    assert primitive.motion_context["held_object_id"] == "object"
    assert not primitive.payload_release_events
    assert primitive.verification_evidence["geometry_confirmation"]["accepted"] is False


def test_accepted_release_rebuilds_same_remaining_goal_and_preserves_watchdog(monkeypatch):
    measured_goal(monkeypatch)
    geometry, monitor = Geometry(), ReleaseGateMonitor()
    primitive = AdaptiveProgramPrimitive(view(gap=.04), {"operations": [operation()]}, geometry,
        interaction_monitor=monitor)
    # Reach the actual final retreat stage, retaining possible payload binding.
    while primitive.stage_index < 7 and not primitive.done:
        advance(primitive, view(gap=.04), view(gap=.04))
    assert primitive.stage_index == 7 and primitive.motion_context["held_object_id"] == "object"
    primitive.stage_goal["xyz_m"][0] += .05
    primitive._watchdog.learned = True
    primitive._new_child(view(gap=.04))
    original_goal = deepcopy(primitive.stage_goal)
    original_limit = primitive._watchdog.limit
    monitor.release_status = "supported"
    advance(primitive, view(gap=.04), view(.001, gap=.04))
    assert primitive.motion_context["held_object_id"] is None
    assert primitive.stage_goal == original_goal
    assert primitive.stage_actions == 1
    assert primitive._watchdog.limit <= original_limit
    assert primitive.payload_release_events[-1]["remaining_watchdog_steps"] == primitive._watchdog.limit-1
    assert primitive.child.motion_context["held_object_id"] is None
    assert not primitive.completed_operations


def test_operation_boundary_guard_survives_continuation_metadata(monkeypatch):
    measured_goal(monkeypatch)
    geometry, monitor = Geometry(), ReleaseGateMonitor()
    primitive = AdaptiveProgramPrimitive(view(gap=.04), {"operations": [operation(), operation("second", "other")]},
        geometry, interaction_monitor=monitor)
    drive_until_pause(primitive)
    saved = primitive.export_continuation()
    saved.update(stage_index=8, phase="move", stage_goal=None, child_continuation=None,
        stage_consumed_steps=0, motion_context=deepcopy(saved["program"]["stages"][8]["motion_context"]))
    resumed = AdaptiveProgramPrimitive.from_continuation(view(gap=.04), saved, geometry,
        interaction_monitor=monitor)
    assert resumed.done and resumed.reason == "program_previous_operation_unverified"
    assert resumed.actions_executed == 0
    assert resumed.progress()["missing_evidence_requests"] == ["accepted_release:first"]
