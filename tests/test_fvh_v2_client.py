"""V2 candidate queue boundary; synthetic images and offline transport only."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from gpt_field.astra import SchemaError
from gpt_field.foundation_vision_harness_V2.client import (
    FoundationVisionHarnessClient, INSTRUCTIONS, harness_schema, validate_contract, ProtocolRepairError,
)
from gpt_field.foundation_vision_harness_V2.protocol import compact_plan
from test_fvh_v1_client import target, view


def candidate(skill="move_to_pose"):
    return {"skill": skill, "xyz_m": [.2, -.1, 1.],
            "quaternion_wxyz": [1., 0., 0., 0.], "gripper": "keep",
            "profile": "transit", "max_steps": 30}


def segment(ident="approach"):
    return {"id": ident, "object": "o1", "anchor": "r1", "contact_mode": "free",
            "contact_parts": [], "keep_upright": True, "checkpoint": "none",
            "candidates": [candidate()]}


def answer(frame=7):
    return {"status": "plan", "why": "Approach the selected visible material.",
            "targets": [target(frame=frame), target("r1", "region", "o1", frame)],
            "segments": [segment()], "completion_conditions": ["The object is visibly held."]}


class OfflineTransport:
    offline_mock = True

    def __init__(self):
        self.calls = []

    def __call__(self, endpoint, payload, headers, timeout_s):
        self.calls.append(deepcopy(payload))
        fields = json.loads(payload["input"][-1]["content"][0]["text"])
        return {"id": f"fixture-{len(self.calls)}", "status": "completed", "model": payload["model"],
                "reasoning": {"effort": "high"}, "offline_mock": True,
                "output": [{"type": "message", "role": "assistant", "status": "completed",
                            "content": [{"type": "output_text", "text": json.dumps(answer(fields["frame_id"]))}]}],
                "usage": {"input_tokens": 0, "output_tokens": 5,
                          "output_tokens_details": {"reasoning_tokens": 2}}}


def test_queue_transport_preserves_current_rgb_and_new_history_protocol():
    transport = OfflineTransport()
    client = FoundationVisionHarnessClient(task_prompt="Grasp the selected object.", transport=transport)
    observed = view()
    for frame in range(2):
        assert client.request_harness(observed, {"step": frame, "frame_id": frame,
            "queue_remaining": 0, "segment_history": [],
            "feedback": {"reason": "selected_region_visible"}}) == validate_contract(answer(frame))
    payload = transport.calls[-1]
    assert payload["text"]["format"]["schema"] == harness_schema()
    assert payload["reasoning"] == {"effort": "high"}
    assert payload["instructions"] == INSTRUCTIONS
    history = json.loads(payload["input"][0]["content"])
    assert history["historical_contract_frame"] == 0 and history["segments"]
    assert "operations" not in history and "targets" not in history
    images = [part for message in payload["input"] if isinstance(message.get("content"), list)
              for part in message["content"] if part.get("type") == "input_image"]
    assert len(images) == 2
    stats = client.history_stats()
    assert stats["calls_by_kind"]["harness"] == stats["accepted_by_kind"]["harness"] == 2
    assert client._history == []
    assert "obstacle_map" not in harness_schema()["properties"]


def test_queue_has_independent_local_budgets_and_retains_target_lineage():
    data = answer()
    data["segments"] = [segment(f"move-{index}") for index in range(8)]
    for item in data["segments"]:
        item["candidates"][0].update(skill="move_relative", xyz_m=[.01, 0., 0.],
                                      quaternion_wxyz=[2., 0., 0., 0.], max_steps=120)
    known = {item["id"]: item for item in data["targets"]}
    data["targets"] = []
    result = validate_contract(data, known_ids=known, frame_id=8, image_shape=(512, 512))
    assert len(result["segments"]) == 8
    assert result["segments"][0]["candidates"][0]["quaternion_wxyz"] == [1., 0., 0., 0.]
    data.update(status="finish", segments=[])
    assert validate_contract(data, known_ids=known)["status"] == "finish"


def test_stale_regions_and_gripper_semantics_are_rejected():
    stale = answer()
    stale["targets"][1]["frame_id"] = 6
    with pytest.raises(SchemaError, match="frame_id"):
        validate_contract(stale, frame_id=7, image_shape=(512, 512))
    ambiguous = answer()
    ambiguous["segments"][0]["candidates"] = [
        {**candidate(), "skill": "set_gripper", "xyz_m": None,
         "quaternion_wxyz": None, "gripper": command} for command in ("open", "close")]
    with pytest.raises(SchemaError, match="semantics"):
        validate_contract(ambiguous)
    fixture = answer()
    fixture["segments"][0].update(contact_mode="articulated", contact_parts=["pad"], checkpoint="held")
    with pytest.raises(SchemaError, match="fixture"):
        validate_contract(fixture)


def test_private_evaluator_state_is_rejected_before_transport():
    transport = OfflineTransport()
    client = FoundationVisionHarnessClient(task_prompt="Grasp the selected object.", transport=transport)
    with pytest.raises(ValueError, match="private evaluator"):
        client.request_harness(view(), {"step": 0, "frame_id": 0,
            "feedback": {"evidence": {"libero_success": True}}})
    assert transport.calls == []


def test_robot_parent_hints_normalize_without_relaxing_material_or_action_identity():
    data = answer()
    data["targets"] += [target("arm", "robot", None, 7), target("finger", "robot", "arm", 7)]
    wire = compact_plan(data)
    normalized = validate_contract(wire, frame_id=7, image_shape=(512, 512))
    by_id = {item["id"]: item for item in normalized["targets"]}
    assert by_id["finger"]["parent_id"] is None
    assert by_id["r1"]["parent_id"] == "o1"
    assert wire["targets"][-1]["parent_id"] == "arm"
    # Re-supplying a redundant robot parent must not look like identity rebinding.
    assert validate_contract(wire, known_ids=by_id, frame_id=7)["targets"] == normalized["targets"]
    for parent in ("o1", "not_observed", "finger"):
        invalid = deepcopy(wire)
        invalid["targets"][-1]["parent_id"] = parent
        with pytest.raises(ProtocolRepairError, match="only region"):
            validate_contract(invalid, frame_id=7)
    cycle = deepcopy(wire)
    cycle["targets"][-2]["parent_id"] = "finger"
    with pytest.raises(ProtocolRepairError, match="only region"):
        validate_contract(cycle, frame_id=7)
    wrong_region = deepcopy(wire)
    wrong_region["targets"][1]["parent_id"] = "arm"
    with pytest.raises(ProtocolRepairError, match="region parent"):
        validate_contract(wrong_region, frame_id=7)
    robot_action = deepcopy(wire)
    robot_action["segments"][0]["object"] = "finger"
    with pytest.raises(ProtocolRepairError, match="robot segmentation"):
        validate_contract(robot_action, frame_id=7)


def test_release_checkpoint_inherits_state_object_without_contact_permission():
    data = answer()
    release = segment("release")
    release.update(contact_mode="grasp", contact_parts=["finger", "pad"])
    release["candidates"] = [{"skill": "set_gripper", "xyz_m": None, "quaternion_wxyz": None,
                              "gripper": "open", "profile": "contact", "max_steps": 16}]
    retreat = segment("retreat")
    retreat.update(object=None, checkpoint="released")
    data["segments"] = [release, retreat]
    normalized = validate_contract(data)
    assert normalized["segments"][1]["object"] is None
    assert normalized["segments"][1]["contact_parts"] == []
    assert normalized["segments"][1]["checkpoint_object"] == "o1"
    data["segments"] = [retreat]
    with pytest.raises(ProtocolRepairError, match="unresolved"):
        validate_contract(data)
    assert validate_contract(data, released_object_id="o1")["segments"][0]["checkpoint_object"] == "o1"
    data["segments"][0]["checkpoint_object"] = "r1"
    with pytest.raises(ProtocolRepairError, match="conflicts"):
        validate_contract(data, released_object_id="o1")


def test_compact_grounded_position_keeps_null_until_measured_projection():
    expanded = validate_contract(answer())
    data = compact_plan(expanded)
    reference = {"region": "r1", "view": "external", "pixel": [100., 110.],
                 "offset_hand_m": [0., 0., -.0036], "standoff_m": .01, "reference": "tool"}
    data["segments"][0]["grounding"] = reference
    data["segments"][0]["candidates"][0]["xyz_m"] = None
    normalized = validate_contract(data, frame_id=7, image_shape=(512, 512))
    part = normalized["segments"][0]
    assert part["id"] == "segment_01" and part["grounding"]["frame_id"] == 7
    assert part["candidates"][0]["xyz_m"] is None
    assert part["candidates"][0]["max_steps"] == 120
    assert normalized["targets"][0]["frame_id"] == 7
    data["segments"][0]["grounding"] = None
    with pytest.raises(ProtocolRepairError, match="xyz_m"):
        validate_contract(data, frame_id=7)
    data["segments"][0]["grounding"] = reference
    data["segments"][0]["candidates"][0].update(skill="move_relative", xyz_m=[.01, 0., 0.])
    with pytest.raises(ProtocolRepairError, match="absolute"):
        validate_contract(data, frame_id=7)


def test_repair_evidence_and_stream_timing_survive_shared_transport_error(tmp_path):
    class InvalidTransport(OfflineTransport):
        def __call__(self, *args):
            response = super().__call__(*args)
            draft = answer(0)
            draft["segments"][0].update(object=None, checkpoint="released")
            response["output"][0]["content"][0]["text"] = json.dumps(draft)
            request_path = next(tmp_path.glob("*.request.json"))
            stream_path = Path(str(request_path).replace(".request.json", ".stream.json"))
            stream_path.write_text(json.dumps({"first_visible_output_s": 2., "last_visible_output_s": 3.,
                "elapsed_s": 3.1, "headers_received_s": .5, "completion_s": 3.05}))
            return response
    client = FoundationVisionHarnessClient(task_prompt="Grasp the selected object.",
                                           transport=InvalidTransport(), log_dir=tmp_path)
    with pytest.raises(ProtocolRepairError) as caught:
        client.request_harness(view(), {"step": 0, "frame_id": 0})
    assert "segments" in caught.value.evidence["draft_json"]
    assert caught.value.evidence["no_candidate_executed"] is True
    timing = client.last_harness_timing
    assert timing["status"] == "protocol_repair" and timing["client_elapsed_s"] > 0
    assert timing["first_visible_output_s"] == 2. and timing["output_generation_s"] == 1.
    assert timing["transport_elapsed_s"] == 3.1 and timing["schema_validation_s"] > 0
    assert client.history_stats()["accepted_by_kind"]["harness"] == 0
