"""Offline geometry, observable-image checks, and versioned client contracts."""

import copy
import json
import math
from unittest.mock import Mock

import numpy as np
import pytest

from gpt_field.astra import SchemaError
from gpt_field.compact_field import expand_compact_field
from gpt_field.compact_subscription import ChatGPTPlanCompactClient, compact_instructions, compact_output_schema
from gpt_field.field_grounding import (
    GROUNDED_FIELD_VERSION, VisualGoalConsistencyChecker, effective_compact_field,
    grounded_compact_field_schema, validate_grounded_compact_field,
)
from gpt_field.hybrid_subscription import HYBRID_GRID
from gpt_field.observation import PolicyObservation, image_to_data_url
from test_compact_subscription import compact_field, hybrid
from test_field_history import FakeTransport, envelope, observation


def reference(kind="held_object", anchor=None):
    return {"kind": kind, "object_center_offset_m": [.04, 0., 0.] if kind == "held_object" else None,
            "desired_object_center_offset_m": [.1, .2, 0.] if kind == "held_object" else None,
            "visual_anchor": anchor}


def grounded_field(ref=None):
    return {**compact_field(), "goal_reference": ref}


def improved_client(fake=None, **updates):
    return ChatGPTPlanCompactClient(token_provider=lambda: "offline-oauth", transport=fake,
        **{"field_improvements": True, "controller_mode": "hybrid", "execution_mode": "tracking",
           "reasoning_effort": "medium", "feedback_history": "block_summary", **updates})


def test_held_object_goal_corrects_translation_and_preserves_raw():
    raw = grounded_field(reference())
    snapshot = copy.deepcopy(raw)
    effective, diagnostics = effective_compact_field(raw, observation())
    assert effective["goal_offset_m"] == pytest.approx([.06, .2, 0.])
    assert "goal_reference" not in effective
    assert diagnostics["raw_goal_offset_m"] == [.05, 0., 0.]
    assert diagnostics["rotation_compensated"]
    assert diagnostics["effective_goal_offset_m"] == pytest.approx([.06, .2, 0.])
    assert raw == snapshot


@pytest.mark.parametrize("current,target,offset,expected", [
    ([0, 0, 0], [0, 0, math.sqrt(.5), math.sqrt(.5)], [.04, 0, 0], [.1, .16, 0]),
    ([0, 0, math.pi/2], [0, 0, 0, 1], [0, .04, 0], [.06, .2, 0]),
    ([0, 0, 0], [0, 0, -math.sqrt(.5), -math.sqrt(.5)], [.04, 0, 0], [.1, .16, 0]),
])
def test_grasp_offset_rotates_from_measured_to_target_orientation(current, target, offset, expected):
    ref = reference()
    ref["object_center_offset_m"] = offset
    raw = grounded_field(ref)
    raw["orientation_target_xyzw"] = target
    obs = observation(state=[.1, .2, 1., *current, .04, -.04])
    effective, _ = effective_compact_field(raw, obs)
    assert effective["goal_offset_m"] == pytest.approx(expected)


@pytest.mark.parametrize("ref", [None, reference("end_effector")])
def test_unspecified_or_end_effector_goal_preserves_raw_intent(ref):
    raw = grounded_field(ref)
    effective, diagnostics = effective_compact_field(raw, observation())
    assert effective == compact_field()
    assert not diagnostics["rotation_compensated"]


@pytest.mark.parametrize("change", [
    {"kind": "simulator_object"}, {"object_center_offset_m": None},
    {"object_center_offset_m": [float("nan"), 0, 0]},
    {"desired_object_center_offset_m": [.51, 0, 0]},
    {"visual_anchor": {"moving_point_uv": [-.01, .5], "target_point_uv": [.5, .5]}},
    {"visual_anchor": {"moving_point_uv": [.5, .5], "target_point_uv": [True, .5]}},
    {"visual_anchor": {"moving_point_uv": [.5, .5], "target_point_uv": [.5, .5], "camera": "wrist"}},
])
def test_grounding_rejects_invalid_or_unbounded_references(change):
    with pytest.raises(SchemaError):
        validate_grounded_compact_field(grounded_field({**reference(), **change}))


def test_derived_goal_outside_allowed_bounds_is_rejected_not_clipped():
    ref = reference()
    ref.update(object_center_offset_m=[-.5, 0, 0], desired_object_center_offset_m=[.5, 0, 0])
    with pytest.raises(SchemaError, match="derived held-object"):
        effective_compact_field(grounded_field(ref), observation())


def test_end_effector_reference_disallows_unused_object_vectors():
    with pytest.raises(SchemaError, match="null"):
        validate_grounded_compact_field(grounded_field({**reference(), "kind": "end_effector"}))


def image_observation(x=50, y=64, step=0, *, duplicate=None, occluded=False, texture=True, wrist_x=None):
    pixels = np.zeros((128, 128, 3), dtype=np.uint8)
    patch = np.random.default_rng(912).integers(0, 256, size=(11, 11, 3), dtype=np.uint8)
    if not occluded and texture:
        pixels[y-5:y+6, x-5:x+6] = patch
        if duplicate is not None:
            pixels[y-5:y+6, duplicate-5:duplicate+6] = patch
    image = image_to_data_url(pixels, size=128, rotate=False)
    wrist = image if wrist_x is None else image_to_data_url(np.roll(pixels, wrist_x-x, axis=1), size=128, rotate=False)
    return PolicyObservation(image, wrist, (.1+.005*step, .2, 1., 0., 0., 0., .04, -.04), "move the bowl")


def visual_reference():
    return reference("end_effector", {"moving_point_uv": [50/127, 64/127],
                                      "target_point_uv": [100/127, 64/127]})


def test_visual_reversal_requires_two_reliable_moving_checks():
    check = VisualGoalConsistencyChecker()
    assert check.configure(image_observation(), visual_reference())["status"] == "configured"
    first = check.observe(image_observation(46, step=1, wrist_x=80))
    second = check.observe(image_observation(42, step=2, wrist_x=90))
    assert not first["refresh_requested"] and first["worsening_checks"] == 1
    assert second["refresh_requested"] and second["reason"] == "visual_divergence"
    assert second["current_distance_px"] == pytest.approx(58)
    assert second["baseline_distance_px"] == pytest.approx(50)


def test_visual_improvement_never_triggers_while_wrist_image_moves_oppositely():
    check = VisualGoalConsistencyChecker()
    check.configure(image_observation(), visual_reference())
    for step, x in enumerate([54, 58, 62], start=1):
        result = check.observe(image_observation(x, step=step, wrist_x=50-4*step))
        assert not result["refresh_requested"] and result["worsening_checks"] == 0


def test_repeated_frames_without_robot_motion_cannot_satisfy_visual_check():
    check = VisualGoalConsistencyChecker()
    check.configure(image_observation(), visual_reference())
    for x in (46, 42):
        result = check.observe(image_observation(x))
        assert result["status"] == "insufficient_robot_motion"
        assert not result["refresh_requested"]


@pytest.mark.parametrize("kwargs,status", [
    ({"texture": False}, "textureless_anchor"), ({"duplicate": 70}, "ambiguous_match"),
])
def test_unreliable_initial_anchor_disables_visual_check(kwargs, status):
    check = VisualGoalConsistencyChecker()
    assert check.configure(image_observation(**kwargs), visual_reference())["status"] == status
    result = check.observe(image_observation(42, step=2))
    assert result["status"] == status and not result["refresh_requested"]


def test_occlusion_or_ambiguity_breaks_required_consecutive_checks():
    check = VisualGoalConsistencyChecker()
    check.configure(image_observation(), visual_reference())
    assert check.observe(image_observation(46, step=1))["worsening_checks"] == 1
    assert check.observe(image_observation(42, step=2, occluded=True))["status"] == "match_lost"
    assert check.observe(image_observation(42, step=3))["worsening_checks"] == 1
    result = check.observe(image_observation(42, step=4, duplicate=60))
    assert result["status"] == "ambiguous_match" and not result["refresh_requested"]


@pytest.mark.parametrize("ref", [None, reference("end_effector")])
def test_null_visual_reference_means_unavailable_not_verified(ref):
    check = VisualGoalConsistencyChecker()
    assert check.configure(image_observation(), ref)["status"] == "visual_anchor_unavailable"
    assert not check.observe(image_observation(42, step=2))["refresh_requested"]


def test_grounded_client_transport_keeps_raw_history_and_expands_corrected_goal(tmp_path):
    raw = hybrid()
    raw["field"] = grounded_field(reference())
    response = envelope(raw, model="gpt-6-astra", reasoning={"effort": "medium"})
    fake = FakeTransport(response)
    instance = improved_client(fake, log_dir=tmp_path)
    result = instance.predict(observation().to_dict())
    assert instance.last_compact_prediction == raw
    assert instance.last_effective_compact_field["goal_offset_m"] == pytest.approx([.06, .2, 0])
    assert result["field"] == expand_compact_field(instance.last_effective_compact_field, HYBRID_GRID)
    assert instance.history_snapshot()[-2:] == response["output"]
    assert fake.calls[0]["reasoning"] == {"effort": "medium"}
    assert instance.budget_snapshot()["api_key_calls"] == 0
    assert instance.field_representation == GROUNDED_FIELD_VERSION
    logged = json.loads(next(tmp_path.glob("*.response.json")).read_text())
    assert json.loads(logged["output"][-1]["content"][0]["text"]) == raw


def test_improved_contact_direct_preserves_five_actions_and_resets_grounding():
    raw = hybrid(True)
    fake = FakeTransport(envelope(raw, reasoning={"effort": "medium"}))
    instance = improved_client(fake)
    result = instance.predict(observation().to_dict())
    assert result["direct"] == raw["direct"]
    assert instance.last_effective_compact_field is None and instance.last_goal_grounding is None


@pytest.mark.parametrize("updates", [{"field_improvements": 1}, {"controller_mode": "direct"},
    {"controller_mode": "field"}, {"execution_mode": "fixed"}, {"reasoning_effort": "high"},
    {"stop_on_model_complete": True}])
def test_improved_mode_rejects_unsupported_configuration_before_auth(updates):
    provider = Mock(return_value="unused")
    options = {"field_improvements": True, "controller_mode": "hybrid", "execution_mode": "tracking",
               "reasoning_effort": "medium", **updates}
    with pytest.raises(ValueError):
        ChatGPTPlanCompactClient(token_provider=provider, transport=FakeTransport(), **options)
    provider.assert_not_called()


def test_improved_payload_is_versioned_and_describes_only_observable_inputs():
    instance = improved_client()
    payload = instance.build_payload(observation().to_dict())
    assert instance.prompt_version.endswith("grounded-v6")
    assert len(payload["text"]["format"]["name"]) <= 64
    assert payload["text"]["format"]["schema"]["properties"]["field"]["anyOf"][0] == grounded_compact_field_schema()
    prompt = payload["instructions"]
    for text in ("at most 30", "at most ten segments", "R_target R_current.T", "FIXED EXTERNAL",
                 "two reliable checks", "five-action Direct", "field_control_event"):
        assert text in prompt
    for outdated in ("Three local plans occur", "only after 30", "all ten ordered", "Reuse the expanded fixed-center field for 30"):
        assert outdated not in prompt
    assert "task_complete" not in json.dumps(payload["text"]["format"]["schema"])


def test_disabled_improvements_preserve_legacy_prompt_and_schema():
    for mode in ("hybrid", "direct", "field"):
        assert compact_instructions(mode) == compact_instructions(mode, field_improvements=False)
        assert compact_output_schema(mode) == compact_output_schema(mode, field_improvements=False)
        assert "goal_reference" not in json.dumps(compact_output_schema(mode))


def test_event_history_preserves_reason_with_only_allowlisted_visual_scalars():
    instance = improved_client()
    event = {"type": "field_control_event", "step": 2, "chunk_index": 0, "reason": "visual_divergence",
             "visual_check": {"status": "tracked", "baseline_distance_px": 50., "current_distance_px": 58.,
                              "moving_point_uv": [.4, .5], "arbitrary_forbidden_key": "not retained"}}
    instance.record_field_event(event)
    record = json.loads(instance.history_snapshot()[-1]["content"][0]["text"])
    assert record["reason"] == "visual_divergence"
    assert set(record["visual_check"]) == {"status", "baseline_distance_px", "current_distance_px"}
    assert instance.history_stats()["feedback_records"] == 0
    assert not instance.last_task_complete


def test_grounding_failure_does_not_commit_raw_response_or_leave_stale_goal():
    raw = hybrid()
    raw["field"] = grounded_field(reference())
    raw["field"]["goal_reference"]["desired_object_center_offset_m"] = [.7, 0, 0]
    instance = improved_client(FakeTransport(envelope(raw, reasoning={"effort": "medium"})))
    with pytest.raises(SchemaError):
        instance.predict(observation().to_dict())
    assert instance.history_snapshot() == []
    assert instance.last_effective_compact_field is None and instance.last_goal_grounding is None
