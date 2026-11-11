"""Contract boundary checks using synthetic RGB and an offline transport only."""
from copy import deepcopy
import json

import numpy as np
import pytest

from gpt_field.astra import SchemaError
from gpt_field.agent_speedup_v2_user_check_sim import CalibratedObservation
from gpt_field.contact_skills import StepView
from gpt_field.foundation_vision_harness_V1.client import (
    FoundationVisionHarnessClient, HISTORY_CONTRACTS, INSTRUCTIONS,
    harness_schema, validate_contract,
)
from gpt_field.observation import image_to_data_url
from test_foundation_vision_bridge import metadata


def target(ident="o1", role="object", parent=None, frame=7):
    return {"id": ident, "query": "surface" if role == "region" else "object",
            "role": role, "parent_id": parent, "frame_id": frame,
            "region_pixels": {"external": [120, 120, 180, 180], "wrist": None}}


def operation(kind="grasp"):
    return {"kind": kind, "object": "o1", "target": None, "grasp_region": "r1",
            "relation": None, "keep_upright": True, "approach_direction": None,
            "jaw_axis": None, "displacement": None, "rotation_axis": None,
            "rotation_angle_rad": None, "offset_m": [0., 0., 0.],
            "contact_parts": ["fingers", "pads"], "stop_on": "goal"}


def answer(frame=7):
    return {"status": "plan", "why": "Use the selected visible material region.",
            "targets": [target(frame=frame), target("r1", "region", "o1", frame)],
            "operations": [operation()], "completion_conditions": ["The selected object is securely held."]}


def view():
    state = (.2, -.1, 1., .2, -.3, .1, .04, -.04)
    url = image_to_data_url(np.zeros((512, 512, 3), np.uint8), size=512, rotate=False)
    obs = CalibratedObservation(url, url, state, "Grasp the selected object.",
                               (0.,)*7, (0.,)*7, geometry=metadata(list(state), 512))
    return StepView(obs)


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


def test_anchored_contract_uses_new_schema_and_current_rgb_only():
    transport = OfflineTransport()
    client = FoundationVisionHarnessClient(task_prompt="Grasp the selected object.", transport=transport)
    observed = view()
    for frame in range(HISTORY_CONTRACTS+2):
        result = client.request_harness(observed, {"step": frame, "frame_id": frame,
            "scene": {"revision": frame, "objects": [], "scene_points": np.zeros((100, 3)),
                      "evidence": {"points_world": np.zeros((100, 3))}},
            "feedback": {"reason": "selected_region_visible"}})
        assert result == answer(frame)
    payload = transport.calls[-1]
    assert payload["text"]["format"]["schema"] == harness_schema()
    assert payload["reasoning"] == {"effort": "high"}
    assert len(payload["input"]) == HISTORY_CONTRACTS+1
    images = [part for message in payload["input"] if isinstance(message.get("content"), list)
              for part in message["content"] if part.get("type") == "input_image"]
    assert len(images) == 2
    text = payload["input"][-1]["content"][0]["text"]
    assert "scene_points" not in text and "points_world" not in text
    assert all("region_pixels" not in message["content"] for message in payload["input"][:-1])
    stats = client.history_stats()
    assert stats["calls_by_kind"]["harness"] == HISTORY_CONTRACTS+2
    assert stats["accepted_by_kind"]["harness"] == HISTORY_CONTRACTS+2
    assert stats["visible_tokens_by_kind"]["harness"] == 3*(HISTORY_CONTRACTS+2)
    assert client._history == []
    assert "keyposes" not in harness_schema()["properties"]["operations"]["items"]["properties"]
    for cue in ("LIBERO", "RoboCasa", "can obstacle", "Simulation pauses", "Only the independent evaluator"):
        assert cue not in INSTRUCTIONS


@pytest.mark.parametrize("change,match", [
    (lambda a: a["targets"][1].update(frame_id=6), "frame_id"),
    (lambda a: a["targets"][1].update(parent_id="unknown"), "parent_id"),
    (lambda a: a["operations"][0].update(grasp_region="o1"), "explicit grasp_region"),
    (lambda a: a["operations"][0].update(target="invented"), "known target"),
    (lambda a: a["operations"][0].update(goal_pose=[0., 0., 1.]), "unexpected fields"),
    (lambda a: a["targets"][0]["region_pixels"].update(external=[0, 0, 513, 40]), "beyond"),
    (lambda a: a["targets"][0].update(frame_id=True), "frame_id"),
])
def test_contract_keeps_region_identity_and_observation_lineage(change, match):
    data = answer()
    change(data)
    with pytest.raises(SchemaError, match=match):
        validate_contract(data, frame_id=7, image_shape=(512, 512))


def test_existing_targets_can_be_referenced_but_not_reparented():
    known = {x["id"]: x for x in answer()["targets"]}
    data = answer()
    data["targets"] = []
    assert validate_contract(data, known_ids=known)["operations"][0]["grasp_region"] == "r1"
    data = answer()
    data["targets"].append(target("o2"))
    data["targets"][1]["parent_id"] = "o2"
    with pytest.raises(SchemaError, match="may not be rebound"):
        validate_contract(data, known_ids=known)


def test_private_evaluator_state_is_rejected_before_transport():
    transport = OfflineTransport()
    client = FoundationVisionHarnessClient(task_prompt="Grasp the selected object.", transport=transport)
    with pytest.raises(ValueError, match="private evaluator"):
        client.request_harness(view(), {"step": 0, "frame_id": 0,
            "feedback": {"evidence": {"libero_success": True}}})
    assert transport.calls == []


def test_contact_rotation_and_terminal_status_are_explicit():
    data = answer()
    data["operations"][0].update(kind="contact_move", rotation_axis=[0., 0., 1.],
                                 rotation_angle_rad=.3, stop_on="contact")
    assert validate_contract(data)["operations"][0]["rotation_angle_rad"] == .3
    data["operations"][0]["rotation_axis"] = None
    with pytest.raises(SchemaError, match="paired"):
        validate_contract(data)
    data = answer()
    data.update(status="blocked", operations=[], targets=[])
    assert validate_contract(data)["status"] == "blocked"
