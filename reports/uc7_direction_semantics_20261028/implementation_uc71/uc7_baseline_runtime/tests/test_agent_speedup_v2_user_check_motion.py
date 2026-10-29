"""One focused protocol check set; no model calls, archived run data or simulator."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from gpt_field.agent_speedup_v2_plan import PlanParseError, repair_schema
from gpt_field.agent_speedup_v2_user_check_motion import (
    LocalMotionAnswer, LocalMotionPrimitive, conform_reviewed_repair,
    motion_context, parse_reviewed_repair, reviewed_repair_schema,
)
from gpt_field.contact_skills import StepView


def _view(x=0.1, y=-0.1, z=0.9):
    return StepView(SimpleNamespace(state=[x, y, z, np.pi, 0., 0., .04, -.04]))


def _motion(**changes):
    return {"frame": "world", "position_mode": "relative", "xyz_m": [0., 0., .04],
            "quaternion_wxyz": None, "gripper": "keep", "steps": 2, **changes}


def _answer(motion=None):
    return {"why": "Lift 4 cm in world +z, then inspect fresh images.", "action": "local_motion",
            "part": None, "step": _motion() if motion is None else motion}


def test_extended_schema_preserves_native_repair_and_rejects_action_payload_mismatch():
    native = repair_schema()
    extended = reviewed_repair_schema()
    assert "local_motion" not in native["properties"]["action"]["enum"]
    assert "local_motion" in extended["properties"]["action"]["enum"]
    answer = parse_reviewed_repair(_answer())
    assert isinstance(answer, LocalMotionAnswer) and answer.to_json() == _answer()
    retry = {"why": "Retry from the current pose.", "action": "retry", "part": None, "step": None}
    assert parse_reviewed_repair(retry).to_json() == retry
    for bad in ({**_answer(), "action": "retry"}, {**_answer(), "part": "A"},
                {**_answer(), "step": None}, {k: v for k, v in _answer().items() if k != "step"}):
        with pytest.raises(PlanParseError):
            conform_reviewed_repair(bad)
    for bad_motion in (_motion(xyz_m=[True, 0., 0.]), _motion(xyz_m=[float("nan"), 0., 0.]),
                       _motion(quaternion_wxyz=[0., 0., 0., 0.]), _motion(steps=6), _motion(frame="camera")):
        with pytest.raises(PlanParseError):
            parse_reviewed_repair(_answer(bad_motion))


def test_relative_segment_is_resolved_once_and_acknowledged_measured_feedback_is_preserved():
    before = _view()
    primitive = LocalMotionPrimitive(before, _motion(), keep_gripper="close")
    first = primitive.next_action(before)
    assert first[:3] == pytest.approx([0., 0., .4]) and first[-1] == 1.
    middle = _view(z=.91)
    primitive.observe_transition(before, first, middle)
    second = primitive.next_action(middle)
    assert second[:3] == pytest.approx([0., 0., .4])
    final = _view(z=.92)
    primitive.observe_transition(middle, second, final)
    out = primitive.outcome()
    assert primitive.done and primitive.actions_executed == 2
    assert out["reason"] == "local_motion_segment_complete"
    assert out["evidence"]["requested_target"][:3] == pytest.approx([.1, -.1, .94])
    assert out["evidence"]["measured_pose"][:3] == pytest.approx([.1, -.1, .92])
    assert out["evidence"]["position_error_m"] == pytest.approx(.02)
    assert "does not establish arrival, grasp or task success" in out["evidence"]["detail"]


def test_absolute_world_height_and_explicit_gripper_match_relative_command():
    view = _view()
    relative = LocalMotionPrimitive(view, _motion(steps=1), keep_gripper="open")
    absolute = LocalMotionPrimitive(view, _motion(position_mode="absolute", xyz_m=[.1, -.1, .94],
                                                  gripper="open", steps=1))
    assert relative.next_action(view) == pytest.approx(absolute.next_action(view))
    with pytest.raises(ValueError, match="runtime command"):
        LocalMotionPrimitive(view, _motion())


def test_out_of_envelope_target_returns_zero_action_rejection_and_fresh_pose_context():
    view = _view()
    motion = _motion(xyz_m=[0., 0., .06], gripper="close")
    original = deepcopy(motion)
    primitive = LocalMotionPrimitive(view, motion)
    assert primitive.done and primitive.actions_executed == 0
    out = primitive.outcome()
    assert out["status"] == "failed" and out["reason"] == "local_motion_target_rejected"
    assert out["evidence"]["executed_steps"] == 0
    assert "no arm or gripper action was executed" in out["evidence"]["detail"]
    assert motion == original
    context = motion_context(view)
    assert context["xyz_m"] == [.1, -.1, .9]
    assert context["axes"]["z"] == "up"
    assert context["limits"]["max_target_translation_m"] == .05
    extreme = LocalMotionPrimitive(view, _motion(xyz_m=[1.7e308] * 3, gripper="close"))
    assert extreme.done and extreme.outcome()["evidence"]["executed_steps"] == 0
