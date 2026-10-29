"""Focused cross-route lifecycle regression; synthetic controller fixtures only.

Install beside the existing tests after applying the two draft runtime files.
No model, simulator, or real-world-clearance claim is involved.
"""
from copy import deepcopy

import pytest

from gpt_field.agent_speedup_v3_operation import OperationRuntime
from gpt_field.agent_speedup_v3_fragments import (
    AdaptiveFragmentSequencePrimitive, AdaptiveLocalMotionPrimitive,
)
from test_agent_speedup_v3_operation import Geometry as LifecycleGeometry, Monitor, settle
from test_agent_speedup_v3_execution import view, segment, motion, assessment, advance


class Geometry(LifecycleGeometry):
    def __init__(self):
        super().__init__()
        self.calls = []
        self.commands = []
        self.command_blocker = None

    def assess_motion(self, measured, xyz, wxyz, context, step):
        self.calls.append(deepcopy(context))
        return assessment(status="supported", suggested_steps=None)

    def verify_command(self, measured, action, context, step, *, allow_unknown):
        self.commands.append(deepcopy(context))
        return {"command_allowed": self.command_blocker is None,
                "missing_evidence_requests": [] if self.command_blocker is None else [self.command_blocker]}


def setup():
    geometry = Geometry()
    runtime = OperationRuntime(geometry, monitor=Monitor(geometry))
    measured = view(gap=.03)
    context = {"intent": "grasp", "target_id": "package", "held_object_id": None,
               "allowed_contact_ids": ["package"]}
    return runtime, geometry, measured, context


def private(runtime, context, step=12, budget=20):
    return runtime.geometry_context({**context, "commanded_gripper": "close", "gripper_action": False},
                                    step, budget, allow_grasp_probe=True)


def make_motion(kind, measured, geometry, context, runtime, step=12):
    if kind == "fragment":
        return AdaptiveFragmentSequencePrimitive(measured, {"segments": [segment()]}, geometry, context,
            step=step, keep_gripper="close", operation_runtime=runtime)
    return AdaptiveLocalMotionPrimitive(measured, motion(), geometry, context,
        step=step, keep_gripper="close", operation_runtime=runtime)


def test_null_public_held_id_uses_actual_binding_only_after_acknowledged_settling():
    runtime, geometry, measured, context = setup()
    runtime.before_action(measured, context, "close", 0)
    assert "attachment_probe_authorized" not in private(runtime, context, 0)
    settle(runtime, measured, context, last=11)
    assert "attachment_probe_authorized" not in private(runtime, context, 11)
    settle(runtime, measured, context, first=12, last=12)
    result = private(runtime, context)
    assert result["attachment_probe_authorized"] is True
    assert result["held_object_id"] == result["attachment_probe_target_id"] == "package"
    assert context["held_object_id"] is None and "attachment_probe_authorized" not in context
    assert geometry.entities["package"]["attachment_probe_unconfirmed"] is True
    assert runtime.counts["hold_supported"] == 0
    for changed in ({"intent": "transit"}, {"intent": "place"}, {"target_id": "other"},
                    {"held_object_id": "other"}):
        assert "attachment_probe_authorized" not in private(runtime, {**context, **changed})


def test_ordinary_and_semantic_routes_share_one_nonrenewable_probe_deadline():
    runtime, geometry, measured, context = setup()
    settle(runtime, measured, context)
    first = private(runtime, context, budget=20)
    semantic = runtime.probe_context({**context, "held_object_id": "package"}, 20, 100)
    assert semantic["attachment_probe_start_step"] == 12
    assert semantic["attachment_probe_max_steps"] == 20
    # Inner program chunks must not shorten an already issued semantic window.
    child = runtime.geometry_context({**semantic, "commanded_gripper": "close"}, 20, 2, allow_grasp_probe=True)
    assert child["attachment_probe_max_steps"] == first["attachment_probe_max_steps"]
    geometry.entities["package"]["geometry_revision"] += 1
    settle(runtime, measured, context, first=21, last=21)
    assert "attachment_probe_authorized" not in private(runtime, context, 32, 100)


@pytest.mark.parametrize("kind", ["fragment", "direct"])
def test_actual_motion_checks_same_private_probe_at_entry_ack_and_emitted_command(kind):
    runtime, geometry, measured, context = setup()
    settle(runtime, measured, context)
    primitive = make_motion(kind, measured, geometry, context, runtime)
    assert not primitive.done
    advance(primitive, measured, view(.001, gap=.03))
    assert geometry.calls and len(geometry.commands) >= 2
    for row in geometry.calls + geometry.commands:
        assert row["attachment_probe_authorized"] is True
        assert row["held_object_id"] == "package"
        assert row["attachment_probe_start_step"] == 12
    assert primitive.motion_context["held_object_id"] is None


@pytest.mark.parametrize("kind", ["fragment", "direct"])
def test_private_probe_does_not_override_unresolved_final_command_coverage(kind):
    runtime, geometry, measured, context = setup()
    settle(runtime, measured, context)
    geometry.command_blocker = "sweep_coverage:package:0:local_region:z:negative"
    primitive = make_motion(kind, measured, geometry, context, runtime)
    assert primitive.done and primitive.reason == "geometry_predicted_block"
    assert primitive.actions_executed == 0
    assert geometry.commands[-1]["attachment_probe_authorized"] is True
    assert runtime.counts["hold_supported"] == 0


def test_same_fragment_close_to_motion_gets_probe_without_model_handoff():
    runtime, geometry, measured, context = setup()
    primitive = AdaptiveFragmentSequencePrimitive(measured, {"segments": [
        segment("set_gripper", gripper="close", max_steps=20), segment()]}, geometry, context,
        operation_runtime=runtime)
    for _ in range(12):
        advance(primitive, measured, measured)
    assert not primitive.done and primitive.progress()["current_segment_index"] == 1
    assert geometry.calls[-1]["attachment_probe_authorized"] is True
    assert geometry.commands[-1]["attachment_probe_authorized"] is True
    assert runtime.counts["hold_supported"] == 0


def test_only_accepted_actual_release_clears_old_held_declaration_in_private_context():
    runtime, geometry, measured, context = setup()
    settle(runtime, measured, context)
    retreat = {"intent": "retreat", "target_id": "container", "held_object_id": "package",
               "commanded_gripper": "open", "gripper_action": False}
    runtime.before_action(measured, retreat, "open", 12)
    assert runtime.geometry_context(retreat, 12, 20)["held_object_id"] == "package"
    geometry.raw_status, geometry.accept = "supported", False
    runtime.observe_transition(measured, measured, retreat, "open", 13)
    assert runtime.geometry_context(retreat, 13, 20)["held_object_id"] == "package"
    geometry.accept = True
    runtime.observe_transition(measured, measured, retreat, "open", 14)
    assert runtime.geometry_context(retreat, 14, 20)["held_object_id"] is None
    assert retreat["held_object_id"] == "package"
    assert "package" in geometry.entities and geometry.entities["package"]["attachment"] == "none"
    assert runtime.counts["release_supported"] == 1


def test_rejected_new_object_close_discards_prepared_unissued_reference():
    runtime, geometry, measured, context = setup()
    settle(runtime, measured, context)
    runtime.before_action(measured, context, "open", 12)
    runtime.observe_transition(measured, measured, context, "open", 13)
    old_operation = runtime.operation_id
    runtime.before_action(measured, context, "close", 13)
    assert runtime._pending is not None
    next_object = {**context, "target_id": "other"}
    refused = runtime.before_action(measured, next_object, "close", 13)
    assert refused["reason"] == "active_manipulated_object_binding_conflict"
    assert runtime._pending is None
    assert runtime.bound_object_id == "package" and runtime.operation_id == old_operation
    assert runtime.last_command == "open" and runtime.monitor.active["kind"] == "release"
    assert runtime.counts["references_started"] == 2
