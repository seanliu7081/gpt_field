"""Planner output bounds and prompt consistency; offline, with no simulator."""
import copy
import json
from pathlib import Path

import pytest

from gpt_field.agent_speedup_v2_client import AgentSpeedUpV2Client, WORKED_EXAMPLE
from gpt_field.agent_speedup_v2_plan import PlanParseError
from gpt_field.agent_speedup_v2_user_check import plan_validation
from gpt_field.agent_speedup_v2_user_check_client import UserCheckV2Client
from gpt_field.agent_speedup_v2_user_check_plan import parse_plan
from gpt_field.agent_speedup_v2_user_check_spatial import validate_spatial_spec
from gpt_field.astra import SchemaError


TASK = "put both the cream cheese box and the butter in the basket"
REQUEST = {"fields": {}, "images": []}


def example_from_payload(payload):
    prefix = "EXAMPLE (other objects; cells illustrative): "
    line = next(x for x in payload["instructions"].splitlines() if x.startswith(prefix))
    return json.loads(line[len(prefix):])


@pytest.mark.parametrize("mode", ["plan", "replan"])
def test_real_payload_example_has_every_required_field_and_roundtrips(mode):
    baseline_before = copy.deepcopy(WORKED_EXAMPLE)
    client = UserCheckV2Client(task_prompt=TASK)
    payload = client.build_payload(REQUEST, mode=mode)
    example = example_from_payload(payload)
    schema = payload["text"]["format"]["schema"]
    assert set(example) == set(schema["required"]) == set(schema["properties"])
    assert client.validate_response(example, client.grid) == example
    assert parse_plan(example).to_json() == example
    assert plan_validation(example, require_native=False)["valid"]
    assert example["regions"] == example["placement_goals"] == []
    assert "Finish immediately after" in payload["instructions"]
    assert "complete requested plan" in payload["instructions"]
    baseline = AgentSpeedUpV2Client(task_prompt=TASK).build_payload(REQUEST, mode=mode)
    assert example_from_payload(baseline) == WORKED_EXAMPLE == baseline_before
    assert "OUTPUT SERIALIZATION" not in baseline["instructions"]


def spatial_plan():
    client = UserCheckV2Client(task_prompt=TASK)
    raw = example_from_payload(client.build_payload(REQUEST, mode="plan"))
    raw["regions"] = [{"id": "F", "support": "B", "anchor": None,
                       "relation": "on", "frame": "world", "bounds_xy_m": None}]
    raw["parts"].append({"id": "F", "text": "shoe box: inner floor region",
                         "move": "fixed", "w": 44, "x": 37})
    raw["placement_goals"] = [{"obj": "A", "ref": "F", "relation": "inside", "tolerance_m": None}]
    raw["steps"][0].update(do="place_on", ref="F")
    return raw


@pytest.mark.parametrize("field", ["regions", "placement_goals"])
def test_wire_and_local_array_limits_agree(field):
    client = UserCheckV2Client(task_prompt=TASK)
    schema = client.build_payload(REQUEST, mode="plan")["text"]["format"]["schema"]
    assert schema["properties"][field]["maxItems"] == 32
    valid = spatial_plan()
    assert not validate_spatial_spec(valid)
    assert client.validate_response(valid, client.grid) == valid
    oversized = copy.deepcopy(valid)
    oversized[field] *= 33
    with pytest.raises(PlanParseError, match="array length"):
        parse_plan(oversized)
    with pytest.raises(SchemaError):
        client.validate_response(oversized, client.grid)


@pytest.mark.parametrize("field,key", [("regions", "id"), ("regions", "support"),
                                      ("regions", "anchor"), ("placement_goals", "obj"),
                                      ("placement_goals", "ref")])
def test_extension_identifiers_cannot_expand_into_unbounded_text(field, key):
    raw = spatial_plan()
    raw[field][0][key] = "unbounded identifier " * 100
    with pytest.raises(PlanParseError):
        parse_plan(raw)


def test_observed_plane_and_nullable_fields_remain_supported():
    raw = spatial_plan()
    raw["regions"][0].update(support="observed_plane", anchor=None,
                             bounds_xy_m=[[-.05, .05], [-.05, .05]])
    assert parse_plan(raw).to_json() == raw
    assert not validate_spatial_spec(raw)


def test_saved_successful_task0_plan_is_preserved_exactly():
    fixture = json.loads((Path(__file__).parent / "fixtures/user_check_task0_plan.json").read_text())
    client = UserCheckV2Client(task_prompt=fixture["task"])
    client.build_payload(REQUEST, mode="plan")
    raw = fixture["plan"]
    assert client.validate_response(raw, client.grid) == raw
    assert parse_plan(raw).to_json() == raw
    assert len(raw["steps"]) == 2
    assert {s["obj"] for s in raw["steps"]} == {"A", "B"}
