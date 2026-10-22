"""Target presence cannot be bypassed by empty obstacle lists or controller changes."""
import copy
import json

import pytest

from gpt_field.astra import SchemaError
from gpt_field.distance_tasks import resolve_distance_task
from gpt_field.visual_field_subscription_v12 import output_schema as schema_v12
from gpt_field.visual_field_subscription_v14 import (
    HISTORY_VERSION, VERSION, VisualFieldSubscriptionClientV14, output_schema, shared_guidance,
)
from test_field_history import FakeTransport, envelope
from test_visual_field_protocol import observation, input_texts
from test_visual_field_v2_protocol import assert_unique_required
from test_visual_field_v7_protocol import current_record, stream_error
from test_visual_field_v12_protocol import prediction as prediction_v12


def prediction(**kwargs):
    value = prediction_v12(**kwargs)
    value.update(target_geometry={"id": value["target_id"], "center_offset_m": [.04, 0, -.08],
        "half_extents_m": [.03, .03, .025], "uncertainty_m": .008, "allowed_contact": "none"},
        contact_alignment=None)
    return value


def alignment(value, update="seed"):
    value["visual_references"]["wrist"] = {"update": "seed", "target_id": value["target_id"],
        "desired_target_uv": [.5, .5], "seed": {"target_anchor_uv": [.6, .55],
            "target_roi_uv": [.5, .45, .7, .65]}}
    value["contact_alignment"] = {"update": update, "target_id": value["target_id"],
        "external": {"target_contact_point_uv": [.7, .6], "tool_pinch_point_uv": [.32, .44]} if update == "seed" else None,
        "wrist": {"target_contact_point_uv": [.6, .55], "tool_pinch_point_uv": [.49, .57]} if update == "seed" else None}
    return value


def payload_refs(value):
    ref = value["visual_references"]["external"]
    ref.update(moving_kind="payload", moving_id="held_object")
    ref["seed"].update(tool_point_uv=[.3, .3], tool_roi_uv=[.25, .25, .35, .35])
    return value


def client(*responses, **kwargs):
    transport = FakeTransport(*(value if isinstance(value, BaseException) else envelope(
        value, model="gpt-6-astra", reasoning={"effort": "medium"}, id=f"v14_{i}")
        for i, value in enumerate(responses)))
    return VisualFieldSubscriptionClientV14(token_provider=lambda: "offline-oauth-fixture",
                                            transport=transport, **kwargs), transport


@pytest.mark.parametrize("controller", ["visual_field", "direct", "gripper"])
@pytest.mark.parametrize("mode", ["transit", "separate", "interaction"])
def test_target_box_is_independent_of_empty_obstacle_lists_and_raw_output_is_unchanged(controller, mode):
    value = prediction(controller=controller, phase="grasp" if controller == "gripper" else "approach", mode=mode)
    before = copy.deepcopy(value)
    instance, transport = client(value, prediction())
    result = instance.predict(observation().to_dict())
    assert result["target_geometry"] == before["target_geometry"]
    assert result["motion_context"]["nearby_obstacles"] == []
    assert value == before
    instance.predict(observation(1).to_dict())
    raw = [json.loads(part["text"]) for message in transport.calls[-1]["input"] if message.get("role") == "assistant"
           for part in message["content"] if part["type"] == "output_text"]
    assert raw == [before]


@pytest.mark.parametrize("case", ["target_missing", "target_wrong", "target_blank", "target_zero_extent", "target_negative_uncertainty",
    "palm_permission", "contact_without_interaction", "null_approach", "null_interaction", "null_close", "null_direct_close",
    "null_field_align", "null_direct_translation", "payload_support_grasp", "payload_support_tool", "payload_same_target",
    "alignment_missing", "alignment_direct", "alignment_wrong_id", "alignment_one_view", "alignment_retain_coordinates",
    "alignment_payload", "alignment_no_wrist", "alignment_outside_image", "forgotten_neighbor", "direct_six", "release_translation"])
def test_explicit_target_contact_does_not_relax_inherited_or_new_constraints(case):
    value = prediction(controller="visual_field", mode="interaction")
    if case == "target_missing": value.pop("target_geometry")
    elif case == "target_wrong": value["target_geometry"]["id"] = "another_object"
    elif case == "target_blank": value["target_geometry"]["id"] = " "
    elif case == "target_zero_extent": value["target_geometry"]["half_extents_m"][0] = 0
    elif case == "target_negative_uncertainty": value["target_geometry"]["uncertainty_m"] = -.01
    elif case == "palm_permission": value["target_geometry"]["allowed_contact"] = "palm"
    elif case == "contact_without_interaction":
        value["motion_context"].update(translation_mode="transit", intended_contact_id=None)
        value["target_geometry"]["allowed_contact"] = "fingers"
    elif case == "null_approach": value = prediction(controller="visual_field"); value["target_geometry"] = None
    elif case == "null_interaction": value["phase"] = "recover"; value["target_geometry"] = None
    elif case == "null_close": value = prediction(controller="gripper", phase="grasp"); value["target_geometry"] = None
    elif case == "null_direct_close":
        value = prediction(phase="recover"); value["target_geometry"] = None; value["direct_actions"][0][6] = 1
    elif case == "null_field_align": value = prediction(controller="visual_field", phase="align"); value["target_geometry"] = None
    elif case == "null_direct_translation":
        value = prediction(phase="grasp"); value["target_geometry"] = None; value["direct_actions"][0][0] = .02
    elif case.startswith("payload_support_") or case == "payload_same_target":
        value = payload_refs(prediction(controller="gripper", phase="release", mode="interaction"))
        value["gripper_intent"]["command"] = "open"
        value["target_geometry"]["allowed_contact"] = "payload_support"
        if case == "payload_support_grasp": value["phase"] = "grasp"
        elif case == "payload_support_tool": value["visual_references"]["external"]["moving_kind"] = "tool"
        else: value["visual_references"]["external"]["moving_id"] = value["target_id"]
    elif case == "alignment_missing": value.pop("contact_alignment")
    elif case.startswith("alignment_"):
        value = alignment(value)
        if case == "alignment_direct":
            other = prediction(); other["contact_alignment"] = value["contact_alignment"]; value = other
        elif case == "alignment_wrong_id": value["contact_alignment"]["target_id"] = "other"
        elif case == "alignment_one_view": value["contact_alignment"]["wrist"] = None
        elif case == "alignment_retain_coordinates": value["contact_alignment"]["update"] = "retain"
        elif case == "alignment_payload": payload_refs(value)
        elif case == "alignment_no_wrist": value["visual_references"]["wrist"] = None
        elif case == "alignment_outside_image": value["contact_alignment"]["wrist"]["tool_pinch_point_uv"][0] = 1.1
    elif case == "forgotten_neighbor": value = prediction(controller="visual_field", present=True); value["field_intent"]["obstacles"] = []
    elif case == "direct_six": value = prediction(); value["direct_actions"].append([0.] * 7)
    elif case == "release_translation": value = prediction(phase="release"); value["direct_actions"][0][0] = .01
    before = copy.deepcopy(value)
    instance, transport = client()
    with pytest.raises(SchemaError): instance.validate_response(value, None)
    assert not transport.calls and value == before


@pytest.mark.parametrize("controller,phase", [("visual_field", "recover"), ("direct", "recover"), ("direct", "align"), ("gripper", "release")])
def test_unavailable_target_allows_explicit_recovery_or_held_opening_without_inventing_geometry(controller, phase):
    value = prediction(controller=controller, phase=phase)
    value["target_geometry"] = None
    if controller == "direct": value["direct_actions"] = [[0.] * 7 for _ in range(5)]
    if controller == "gripper": value["gripper_intent"]["command"] = "open"
    instance, _ = client()
    assert instance.validate_response(value, None)["target_geometry"] is None


@pytest.mark.parametrize("update", ["seed", "retain"])
def test_two_view_alignment_is_preserved_for_runtime_validation_not_reinvented(update):
    value = alignment(prediction(controller="visual_field", mode="interaction"), update)
    value["target_geometry"]["allowed_contact"] = "fingers"
    instance, _ = client(value)
    result = instance.predict(observation().to_dict())
    assert result["contact_alignment"] == value["contact_alignment"]
    assert result["target_geometry"]["allowed_contact"] == "fingers"


def test_explicit_payload_support_never_invents_a_palm_exemption():
    value = payload_refs(prediction(controller="visual_field", phase="release", mode="interaction"))
    value["field_intent"].update(motion_axes="z", goal_offset_m=[0, 0, -.01])
    value["target_geometry"]["allowed_contact"] = "payload_support"
    instance, _ = client(value)
    assert instance.predict(observation().to_dict())["target_geometry"] == value["target_geometry"]


def test_bounded_semantic_and_stream_repairs_preserve_same_observation_and_actual_request_count(tmp_path):
    invalid = prediction(); invalid["target_geometry"] = None
    good = prediction()
    instance, transport = client(invalid, stream_error(), good, log_dir=tmp_path)
    assert instance.predict(observation().to_dict()) == good
    assert len(transport.calls) == instance.budget_snapshot()["api_calls"] == 3
    assert instance.budget_snapshot()["api_key_calls"] == 0
    assert instance.history_stats()["observations"] == 1
    assert instance.history_stats()["semantic_repair_requests"] == instance.history_stats()["stream_retry_requests"] == 1
    assert json.loads(current_record(transport.calls[1])["observable_context"]["output_repair"]["rejected_model_output_json"]) == invalid
    assert len(list(tmp_path.glob("*.request.json"))) == 3


@pytest.mark.parametrize("task", ["bowl_plate", "drawer_bowl"])
def test_schema_delta_small_prompt_generic_and_subscription_and_history_preserved(task):
    instance, transport = client(*(prediction() for _ in range(4)), task_key=task)
    for i in range(4): instance.predict(observation(i, task=task).to_dict())
    assert_unique_required(output_schema())
    legacy = output_schema()
    for key in ("target_geometry", "contact_alignment"):
        legacy["properties"].pop(key); legacy["required"].remove(key)
    assert legacy == schema_v12()
    payload = transport.calls[-1]
    assert payload["text"]["format"]["name"] == "visual_field_rgb_joints_v14"
    assert payload["text"]["format"]["schema"] == output_schema()
    assert payload["model"] == "gpt-6-astra" and payload["reasoning"] == {"effort": "medium"}
    assert payload["stream"] and not payload["store"]
    assert instance.prompt_version == VERSION + "-hybrid"
    assert instance.history_stats()["protocol"] == HISTORY_VERSION
    assert len([r for r in input_texts(payload) if r["type"] == "observation"]) == 2
    text = shared_guidance(resolve_distance_task(task))
    for phrase in ("NEVER hand/palm", "independently of other obstacle lists", "actual open-pad midpoint",
                   "not wrist image center", "useful5–8cm goal", "Null permits only finite", "zero XYZ",
                   "actual_segment_must_be_rechecked", "NEVER renews debt", "Current free-image Direct"):
        assert phrase in text
    other = resolve_distance_task("drawer_bowl" if task == "bowl_plate" else "bowl_plate")
    assert text.replace(resolve_distance_task(task).prompt, "TASK") == shared_guidance(other).replace(other.prompt, "TASK")


@pytest.mark.parametrize("key", ["object_poses", "contacts", "depth", "libero_success", "camera_calibration"])
def test_new_geometry_feedback_does_not_admit_hidden_inputs(key):
    instance, transport = client(prediction())
    instance.observable_context = {"visual_feedback": {"target_geometry": {key: "forbidden"}}}
    with pytest.raises(ValueError): instance.predict(observation().to_dict())
    assert not transport.calls


@pytest.mark.parametrize('phase', ['recover', 'release'])
def test_contact_alignment_cannot_retarget_recovery_or_release_toward_grasp(phase):
    value = alignment(prediction(controller='visual_field', phase=phase))
    if phase == 'release': value['field_intent'].update(motion_axes='z', goal_offset_m=[0.,0.,-.01])
    instance, transport = client()
    with pytest.raises(SchemaError, match='approach or align'):
        instance.validate_response(value, None)
    assert not transport.calls
