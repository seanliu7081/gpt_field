"""Inventory is the only scene geometry source; compatibility caps never control."""
import copy
import json

import pytest

from gpt_field.astra import SchemaError
from gpt_field.distance_tasks import resolve_distance_task
from gpt_field.visual_field_subscription_v16 import (
    VisualFieldSubscriptionClientV16, output_schema, shared_guidance, HISTORY_VERSION,
)
from test_field_history import FakeTransport, envelope
from test_visual_field_protocol import observation
from test_visual_field_v14_protocol import prediction as legacy_prediction
from test_visual_field_v7_protocol import current_record, stream_error


def prediction(**kwargs):
    value = legacy_prediction(**kwargs)
    geometry = value.pop("target_geometry")
    value["target_contact"] = geometry.pop("allowed_contact")
    identity = geometry.pop("id")
    value["scene_inventory"] = [{"id": identity, "external_roi_uv": [.6, .5, .8, .7],
        "wrist_roi_uv": None, "geometry": geometry}]
    value["motion_context"].pop("nearby_obstacles")
    value["motion_context"].pop("obstacle_assessment")
    if value["field_intent"] is not None:
        value["field_intent"].pop("obstacles")
    return value


def neighbor(identity="distractor", geometry=True, external=True):
    return {"id": identity, "external_roi_uv": [.1, .1, .2, .2] if external else None,
        "wrist_roi_uv": None if external else [.7, .7, .9, .9],
        "geometry": {"center_offset_m": [.1, -.1, -.1], "half_extents_m": [.02]*3,
                     "uncertainty_m": .03} if geometry else None}


def client(*responses, **kwargs):
    transport = FakeTransport(*(value if isinstance(value, BaseException) else envelope(
        value, model="gpt-6-astra", reasoning={"effort": "medium"}, id=f"v16_{i}")
        for i, value in enumerate(responses)))
    return VisualFieldSubscriptionClientV16(token_provider=lambda: "offline-oauth", transport=transport, **kwargs), transport


@pytest.mark.parametrize("controller", ["visual_field", "direct", "gripper"])
def test_inventory_materializes_one_geometry_source_all_controllers_and_exact_raw_history(controller):
    value = prediction(controller=controller, phase="grasp" if controller == "gripper" else "approach")
    value["scene_inventory"].append(neighbor())
    before = copy.deepcopy(value)
    instance, transport = client(value, prediction())
    result = instance.predict(observation().to_dict())
    assert result["raw_wire_prediction"] == before == value
    assert result["scene_inventory"] == before["scene_inventory"]
    assert result["target_geometry"]["center_offset_m"] == before["scene_inventory"][0]["geometry"]["center_offset_m"]
    assert result["motion_context"]["nearby_obstacles"] == [{"id": "distractor", "external_roi_uv": [.1,.1,.2,.2]}]
    assert result["motion_context"]["obstacle_assessment"] == "present"
    if controller == "visual_field":
        assert result["field_intent"]["obstacles"][0]["lower_offset_m"] == pytest.approx([.08, -.12, -.12])
    instance.predict(observation(1).to_dict())
    history = [json.loads(p["text"]) for m in transport.calls[-1]["input"] if m.get("role") == "assistant"
               for p in m.get("content", []) if p["type"] == "output_text"]
    assert history == [before]
    assert "raw_wire_prediction" not in history[0]


@pytest.mark.parametrize("case", ["missing_inventory", "duplicate_id", "blank_id", "both_views_null", "bad_roi", "bad_extent",
    "legacy_obstacles", "legacy_target", "legacy_neighbors", "legacy_assessment", "unknown_id_overlap", "missing_target_approach"])
def test_malformed_inventory_and_duplicate_scene_authorities_are_rejected(case):
    value = prediction(controller="visual_field")
    if case == "missing_inventory": value.pop("scene_inventory")
    elif case == "duplicate_id": value["scene_inventory"].append(copy.deepcopy(value["scene_inventory"][0]))
    elif case == "blank_id": value["scene_inventory"][0]["id"] = " "
    elif case == "both_views_null": value["scene_inventory"][0]["external_roi_uv"] = None
    elif case == "bad_roi": value["scene_inventory"][0]["external_roi_uv"] = [.8,.8,.6,.6]
    elif case == "bad_extent": value["scene_inventory"][0]["geometry"]["half_extents_m"][0] = 0
    elif case == "legacy_obstacles": value["field_intent"]["obstacles"] = []
    elif case == "legacy_target": value["target_geometry"] = None
    elif case == "legacy_neighbors": value["motion_context"]["nearby_obstacles"] = []
    elif case == "legacy_assessment": value["motion_context"]["obstacle_assessment"] = "none_relevant"
    elif case == "unknown_id_overlap": value["field_intent"]["unknown_regions"] = [dict(id=value["target_id"], **value["scene_inventory"][0]["geometry"])]
    elif case == "missing_target_approach": value["scene_inventory"] = []
    instance, transport = client()
    with pytest.raises(SchemaError): instance.validate_response(value, None)
    assert not transport.calls


def test_twelve_entries_not_truncated_to_legacy_six_neighbor_or_eight_box_caps():
    value = prediction(controller="visual_field")
    value["scene_inventory"].extend(neighbor(f"physical_{i}") for i in range(11))
    instance, _ = client(value)
    result = instance.predict(observation().to_dict())
    assert len(result["scene_inventory"]) == 12
    assert len(result["field_intent"]["obstacles"]) == len(result["motion_context"]["nearby_obstacles"]) == 11


@pytest.mark.parametrize("geometry,external", [(False,True),(True,False)])
def test_image_only_or_wrist_only_objects_remain_inventory_uncertainty_not_synthetic_walls(geometry, external):
    value = prediction(controller="visual_field"); value["scene_inventory"].append(neighbor(geometry=geometry,external=external))
    instance, _ = client(value); result = instance.predict(observation().to_dict())
    assert result["scene_inventory"] == value["scene_inventory"]
    assert result["motion_context"]["obstacle_assessment"] == "unobservable"
    assert result["field_intent"]["unknown_regions"] == []
    assert len(result["field_intent"]["obstacles"]) == int(geometry)


@pytest.mark.parametrize("omitted", [False,True])
def test_known_target_geometry_retained_from_current_observable_memory(omitted):
    value = prediction(controller="visual_field")
    known = copy.deepcopy(value["scene_inventory"][0]["geometry"])
    if omitted: value["scene_inventory"] = []
    else: value["scene_inventory"][0]["geometry"] = None
    instance, _ = client(value)
    instance.observable_context = {"visual_feedback": {"scene_inventory": {"entries": [
        {"id":value["target_id"], "current_geometry":known}]}}}
    result = instance.predict(observation().to_dict())
    assert result["target_geometry"]["center_offset_m"] == known["center_offset_m"]
    assert result["raw_wire_prediction"] == value


def test_same_observation_subscription_repairs_and_guidance_do_not_relax_contact():
    bad = prediction(); bad["direct_actions"] = None
    value = prediction(phase="recover"); value["scene_inventory"] = []
    value["direct_actions"] = [[0.]*7 for _ in range(5)]
    instance, transport = client(stream_error(), bad, value)
    result = instance.predict(observation().to_dict())
    assert result["target_geometry"] is None and result["direct_actions"] == value["direct_actions"]
    assert len(transport.calls) == instance.budget_snapshot()["api_calls"] == 3
    assert instance.budget_snapshot()["api_key_calls"] == 0
    assert len({current_record(p)["observation_index"] for p in transport.calls}) == 1
    assert all(p["model"] == "gpt-6-astra" and p["reasoning"] == {"effort":"medium"} for p in transport.calls)
    text = shared_guidance(resolve_distance_task("bowl_plate"))
    assert "ALL currently visible nonrobot" in text and "NEVER hand/palm" in text
    assert "max(30mm, target uncertainty)" in text and "lateral/away recovery remains available" in text
    assert instance.history_stats()["protocol"] == HISTORY_VERSION
    assert output_schema()["properties"]["scene_inventory"]["maxItems"] == 12


def test_remembered_rebased_offsets_are_unbounded_exact_and_validation_proxy_never_escapes():
    value = prediction(controller="visual_field")
    value["scene_inventory"][0]["geometry"] = None
    value["scene_inventory"].append(neighbor(geometry=False))
    remembered = {"center_offset_m": [.71,-.82,.93], "half_extents_m": [.02]*3,"uncertainty_m":.03}
    instance, transport = client(value,prediction())
    instance.observable_context = {"visual_feedback":{"scene_inventory":{"entries":[
        {"id":value["target_id"],"current_geometry":copy.deepcopy(remembered)},
        {"id":"distractor","current_geometry":copy.deepcopy(remembered)}]}}}
    result = instance.predict(observation().to_dict())
    assert result["target_geometry"]["center_offset_m"] == [.71,-.82,.93]
    assert result["field_intent"]["obstacles"][0]["lower_offset_m"] == [.69,-.84,.91]
    assert result["raw_wire_prediction"] == value
    instance.predict(observation(1).to_dict())
    raw = [json.loads(p["text"])for m in transport.calls[-1]["input"]if m.get("role")=="assistant"
           for p in m.get("content",[])if p["type"]=="output_text"]
    assert raw == [value]


def test_unknown_space_cannot_reuse_an_occluded_remembered_physical_object_id():
    value = prediction(controller="visual_field")
    geometry = neighbor()["geometry"]
    value["field_intent"]["unknown_regions"] = [dict(id="remembered_object", **geometry)]
    instance, transport = client()
    instance.observable_context = {"visual_feedback":{"scene_inventory":{"entries":[
        {"id":"remembered_object", "current_geometry":geometry}]}}}
    with pytest.raises(SchemaError, match="distinct from physical inventory"):
        instance.validate_response(value,None)
    assert not transport.calls
