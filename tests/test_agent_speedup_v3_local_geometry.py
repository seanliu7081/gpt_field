"""Goal-local execution: unknown evidence stays unknown, hard failures stay hard."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np

from gpt_field.agent_speedup_v3_local_geometry import LocalGeometryMemory
from gpt_field.agent_speedup_v3_fragments import AdaptiveFragmentSequencePrimitive
from gpt_field.contact_skills import StepView
from test_agent_speedup_v3_geometry import VisualWatchFake, entity, update, context


def view(x=0.):
    return StepView(SimpleNamespace(state=[x, 0., .8, np.pi, 0., 0., .04, -.04],
                                    image="synthetic-public-rgb", wrist_image=None, geometry=None))


def memory():
    return LocalGeometryMemory(visual_watch=VisualWatchFake())


def test_absent_free_region_is_advisory_not_false_free_space():
    result = memory().assess_motion(view(), [.15, 0., .8], None, context(), 0)
    assert result["execution_allowed"]
    assert result["status"] == "unknown"
    assert result["unknown_fraction"] == 1.
    assert not result["physical_clearance_verified"]
    assert result["suggested_steps"] is None
    assert result["missing_evidence_requests"] == []


def test_thin_obstacle_between_endpoints_is_still_blocked():
    m, v = memory(), view()
    m.update_from_model(update(entity("wall", center=(.10, 0, .86), half=(.0002, .15, .2))), v, 0)
    result = m.assess_motion(v, [.2, 0., .8], None, context(), 0)
    assert not result["execution_allowed"]
    assert any("wall" in reason for reason in result["blocking_reasons"])


def test_uncertainty_shell_is_advisory_but_nominal_core_is_guarded():
    m, v = memory(), view()
    # Table top is 0.78, hand is above it. Its inflated shell reaches the hand.
    m.update_from_model(update(entity("table", "support_surface", center=(0, 0, .68),
        half=(1., 1., .10), uncertainty_m=.04)), v, 0)
    result = m.assess_motion(v, [.04, 0, .8], None, context(), 0)
    assert result["execution_allowed"]
    lower = m.assess_motion(v, [0, 0, .60], None, context(), 0)
    assert not lower["execution_allowed"]


def test_tracking_failure_cannot_be_overridden_by_empty_geometry():
    result = memory().assess_motion(view(), [.04, 0, .8], None,
                                   context(tracking={"status": "mismatch"}), 0)
    assert not result["execution_allowed"]
    assert "measured_tracking_mismatch" in result["blocking_reasons"]


def test_missing_payload_shape_and_current_rgb_still_block_motion():
    m, v = memory(), view()
    assert not m.assess_motion(v, [.04, 0, .8], None, context(held_object_id="unknown"), 0)["execution_allowed"]
    m.update_from_model(update(entity("unknown", "object", center=(.3, 0, .8))), v, 0)
    result = m.assess_motion(v, [.04, 0, .8], None, context(held_object_id="unknown"), 0)
    assert "payload_shape_required:unknown" in result["blocking_reasons"]
    v.observation.image = None
    assert "current_public_rgb_required" in m.assess_motion(v, [.04, 0, .8], None, context(), 0)["blocking_reasons"]


def test_exact_command_checks_bounds_and_new_obstacles_despite_cached_route():
    m, v = memory(), view()
    assert m.assess_motion(v, [.10, 0, .8], None, context(), 0)["execution_allowed"]
    assert not m.verify_command(v, [1., 0, 0, 0, 0, 0, -1], context(), 0)["command_allowed"]
    m.update_from_model(update(entity("wall", center=(.02, 0, .86), half=(.0002, .15, .2))), v, 1)
    assert not m.verify_command(v, [.4, 0, 0, 0, 0, 0, -1], context(), 1)["command_allowed"]


def test_local_progress_can_continue_past_old_unknown_horizon_without_model():
    m, before = memory(), view()
    segment = {"skill": "move_relative", "xyz_m": [.04, 0, 0], "quaternion_wxyz": None,
               "gripper": "keep", "profile": "transit", "max_steps": 8, "observation_after": False}
    primitive = AdaptiveFragmentSequencePrimitive(before, {"segments": [segment]}, m, context())
    for step in range(1, 50):
        action = primitive.next_action(before)
        after = view(min(.04, step*.001))
        primitive.observe_transition(before, action, after)
        before = after
        if primitive.done:
            break
    assert primitive.done and primitive.progress()["goal_reached"]
    assert primitive.actions_executed > 8
    assert primitive.progress()["local_continuations"] > 0


def test_preview_does_not_mutate_live_evidence_or_route_accounting():
    m, v = memory(), view()
    before = deepcopy(m.snapshot()), deepcopy(m.local_compute_counts)
    assert m.preview_motion(v, [.04, 0, .8], None, context(), 0)["execution_allowed"]
    assert before == (m.snapshot(), m.local_compute_counts)


def test_newly_relevant_changed_obstacle_is_checked_before_first_action():
    m, v = memory(), view()
    m.update_from_model(update(entity("nearby", center=(.15, .10, .86), half=(.02, .02, .02))), v, 0)
    m.visual_watch.reports["nearby"] = {"status": "changed", "reason": "synthetic_observed_change"}
    result = m.assess_motion(v, [.15, 0, .8], None, context(), 1)
    assert "changed_geometry:nearby" in result["blocking_reasons"]


def test_moving_gripper_command_does_not_bypass_finite_attachment_probe():
    m, v = memory(), view()
    m.update_from_model(update(entity("payload", "object", center=(0, 0, .82), half=(.01, .01, .01))), v, 0)
    m.begin_attachment_probe("payload", v, 0)
    ctx = context("grasp", target_id="payload", held_object_id="payload",
                  allowed_contact_ids=["payload"], gripper_action=True, commanded_gripper="close")
    result = m.assess_motion(v, [.04, 0, .8], None, ctx, 2)
    assert "attachment_verification_required" in result["blocking_reasons"]
