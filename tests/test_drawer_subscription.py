"""Task-specific drawer subscription contracts; offline transport only."""

import copy
from dataclasses import FrozenInstanceError, replace
import hashlib
import json
from unittest.mock import Mock

import pytest

from gpt_field.astra import SchemaError
from gpt_field.direct_history import TASK
from gpt_field.distance_tasks import DISTANCE_TASK_KEYS, resolve_distance_task
from gpt_field.distance_subscription import (
    ChatGPTPlanDistanceClient, control_context_schema, distance_field_version,
    distance_instructions, distance_output_schema, distance_prompt_version, distance_schema_name,
    validate_control_context,
)
from gpt_field.compact_subscription import compact_operational_guidance, compact_output_schema
from gpt_field.hybrid_subscription import HYBRID_GRID, hybrid_output_schema, validate_hybrid_prediction
from test_distance_subscription import prediction, response
from test_field_history import FakeTransport, observation


DRAWER = resolve_distance_task("drawer_bowl")


def drawer_observation(index=0):
    return replace(observation(index), prompt=DRAWER.prompt)


def drawer_prediction(*, contact=False, phase="approach", target="drawer_handle"):
    raw = prediction(contact=contact, phase=phase, visual=False)
    raw["target_region"] = target
    return raw


def drawer_client(transport=None, **kwargs):
    return ChatGPTPlanDistanceClient(task_key="drawer_bowl", transport=transport,
        token_provider=lambda: "drawer-offline-oauth", **kwargs)


def test_task_registry_is_explicit_immutable_and_contains_no_hidden_geometry():
    assert DISTANCE_TASK_KEYS == ("bowl_plate", "drawer_bowl")
    assert (DRAWER.suite, DRAWER.task_id, DRAWER.prompt) == (
        "libero_goal", 3, "open the top drawer and put the bowl inside")
    assert DRAWER.target_regions == ("drawer_handle", "black_bowl", "drawer_interior")
    assert DRAWER.placement_target == "drawer_interior" and not DRAWER.allow_upward_reobservation
    assert {"handle_grasp", "drawer_open", "handle_release"} <= set(DRAWER.control_phases)
    assert "drawer_close" not in DRAWER.control_phases
    assert set(DRAWER.__dataclass_fields__) == {
        "key", "suite", "task_id", "prompt", "target_regions", "placement_target",
        "control_phases", "allow_upward_reobservation"}
    with pytest.raises(FrozenInstanceError):
        DRAWER.prompt = "another task"
    old = resolve_distance_task()
    assert (old.key, old.suite, old.task_id, old.prompt) == ("bowl_plate", "libero_spatial", 0, TASK)
    assert old.allow_upward_reobservation


@pytest.mark.parametrize("key", [None, "drawer", "libero_goal", [], {}, 3])
def test_unknown_task_is_rejected_before_authentication(key):
    provider, transport = Mock(), FakeTransport()
    with pytest.raises(ValueError, match="Unknown distance task"):
        ChatGPTPlanDistanceClient(task_key=key, token_provider=provider, transport=transport)
    provider.assert_not_called()
    assert not transport.calls


def test_existing_bowl_protocol_bytes_and_schemas_are_preserved():
    # Captured before adding the task parameter; a new drawer protocol must not
    # silently rewrite historical bowl instructions or any default schema.
    values = {"instructions": distance_instructions(), "schema": distance_output_schema(),
              "context": control_context_schema(), "compact": compact_output_schema(),
              "hybrid": hybrid_output_schema()}
    assert hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest() == (
        "50458ad290622b4608af65a8cfcb3d52db906e9e4a0c01cb68e1b9b858ff94fa")
    assert distance_instructions() == distance_instructions(task_key="bowl_plate")
    assert distance_output_schema() == distance_output_schema(task_key="bowl_plate")


def test_drawer_preview_uses_new_versions_and_task_specific_phase_route_schema():
    provider, transport = Mock(), FakeTransport()
    client = ChatGPTPlanDistanceClient(task_key="drawer_bowl", token_provider=provider, transport=transport)
    payload = client.build_payload(drawer_observation().to_dict())
    assert client.task_spec == DRAWER and client.task_prompt == DRAWER.prompt
    assert client.target_regions == DRAWER.target_regions
    assert client.prompt_version == distance_prompt_version("drawer_bowl")
    assert client.field_representation == distance_field_version("drawer_bowl")
    assert client.prompt_version != distance_prompt_version()
    assert client.field_representation != distance_field_version()
    assert payload["text"]["format"] == {
        "type": "json_schema", "name": distance_schema_name("drawer_bowl"), "strict": True,
        "schema": distance_output_schema(task_key="drawer_bowl")}
    schema = payload["text"]["format"]["schema"]
    assert schema["properties"]["target_region"]["enum"] == list(DRAWER.target_regions)
    assert schema["properties"]["control_context"]["properties"]["phase"]["enum"] == list(DRAWER.control_phases)
    assert "task_complete" not in schema["properties"]
    assert payload["model"] == "gpt-6-astra" and payload["reasoning"] == {"effort": "medium"}
    assert payload["stream"] is True and payload["store"] is False
    assert client.chunk_size == 5 and client.grid == HYBRID_GRID
    observed = json.loads(payload["input"][-1]["content"][0]["text"])
    assert observed["prompt"] == DRAWER.prompt
    assert not transport.calls
    provider.assert_not_called()


def test_drawer_instructions_distinguish_handle_pull_bowl_lift_and_constrained_release():
    prompt = distance_instructions(task_key="drawer_bowl")
    for required in (DRAWER.prompt, "handle_grasp", "drawer_open", "handle_release", "drawer_interior",
                     "actual handle grasp, pull and release always use exactly five Direct actions",
                     "Do not assume a fixed pull axis", "does NOT call for a vertical test lift",
                     "For the BOWL grasp only", "Closing the drawer is not required",
                     "Automatic upward reobservation is disabled", "no local +Z recovery",
                     "No object poses", "drawer joint positions", "known handle coordinates",
                     "26-neighbor", "Dijkstra", "0.005 m", "120 total actions",
                     "no episode action or time cap", "There is no task_complete output"):
        assert required in prompt
    assert "plate" not in prompt.lower()
    assert compact_operational_guidance(centered_placement_guidance=True, grasp_settle_guidance=True) not in prompt
    assert "at most 2 cm" not in prompt


@pytest.mark.parametrize("phase", ["handle_grasp", "drawer_open", "handle_release"])
def test_drawer_handle_phases_validate_only_in_drawer_context(phase):
    context = {"phase": phase, "visual_reference": None}
    assert validate_control_context(context, task_key="drawer_bowl") == context
    with pytest.raises(SchemaError, match="phase"):
        validate_control_context(context)
    with pytest.raises(SchemaError, match="phase"):
        validate_control_context({**context, "phase": "drawer_close"}, task_key="drawer_bowl")


@pytest.mark.parametrize("phase,target", [
    ("handle_grasp", "drawer_handle"), ("drawer_open", "drawer_handle"),
    ("handle_release", "drawer_handle"), ("grasp", "black_bowl"),
    ("release", "drawer_interior"), ("verify", "drawer_interior"),
])
def test_drawer_direct_transcript_preserves_raw_task_context_and_five_actions(phase, target):
    raw = drawer_prediction(contact=True, phase=phase, target=target)
    raw["direct"]["actions"] = [[.01*i, -.02*i, 0., .01, 0., 0., -1.] for i in range(5)]
    snapshot = copy.deepcopy(raw)
    returned = response(raw)
    transport = FakeTransport(returned)
    client = drawer_client(transport)
    normalized = client.predict(drawer_observation().to_dict())
    assert normalized["target_region"] == target
    assert normalized["direct"]["actions"] == raw["direct"]["actions"]
    assert "control_context" not in normalized
    assert raw == snapshot and client.last_compact_prediction == snapshot
    assert client.last_control_context == raw["control_context"]
    assert client.history_snapshot()[-2:] == returned["output"]
    assert client.budget_snapshot()["api_key_calls"] == 0
    validate_hybrid_prediction(normalized, target_regions=DRAWER.target_regions)
    client.reset_history()
    assert client.task_spec == DRAWER and client.task_prompt == DRAWER.prompt
    assert client.last_control_context is None


@pytest.mark.parametrize("target", DRAWER.target_regions)
def test_free_space_field_expansion_retains_task_specific_route_and_raw_compact_history(target):
    raw = drawer_prediction(target=target)
    returned = response(raw)
    transport = FakeTransport(returned)
    client = drawer_client(transport)
    normalized = client.predict(drawer_observation().to_dict())
    assert normalized["target_region"] == target
    assert len(normalized["field"]["values"]) == 35937
    assert client.last_compact_prediction == raw
    assert '"values"' not in json.dumps(client.history_snapshot())


@pytest.mark.parametrize("contact,phase,target", [
    (False, "handle_grasp", "drawer_handle"), (False, "drawer_open", "drawer_handle"),
    (False, "handle_release", "drawer_handle"), (True, "drawer_open", "black_bowl"),
    (True, "handle_release", "drawer_interior"), (True, "grasp", "drawer_handle"),
    (True, "release", "drawer_handle"), (True, "release", "black_bowl"),
    (True, "approach", "plate"),
])
def test_wrong_drawer_controller_phase_or_target_does_not_commit_history(contact, phase, target):
    raw = drawer_prediction(contact=contact, phase=phase, target=target)
    client = drawer_client(FakeTransport(response(raw)))
    with pytest.raises(SchemaError):
        client.predict(drawer_observation().to_dict())
    assert not client.history_snapshot()
    assert client.last_control_context is None and client.last_compact_prediction is None


def test_bowl_and_drawer_instances_do_not_leak_task_text_or_routes_between_each_other():
    drawer = drawer_client()
    bowl = ChatGPTPlanDistanceClient(token_provider=lambda: "offline", transport=FakeTransport())
    assert drawer.build_payload(drawer_observation().to_dict())["instructions"] == distance_instructions(task_key="drawer_bowl")
    assert bowl.build_payload(observation().to_dict())["instructions"] == distance_instructions()
    for client, wrong in ((drawer, observation()), (bowl, drawer_observation())):
        with pytest.raises(ValueError, match="Task text"):
            client.build_payload(wrong.to_dict())
    assert hybrid_output_schema()["properties"]["target_region"]["enum"] == ["black_bowl", "plate"]
    assert "drawer_open" not in control_context_schema()["properties"]["phase"]["enum"]


def test_conflicting_task_prompt_is_rejected_before_transport():
    with pytest.raises(ValueError, match="task_prompt"):
        drawer_client(task_prompt=TASK)


@pytest.mark.parametrize("regions", [None, [], [""], ["plate", "plate"], "plate", ["plate", None]])
def test_invalid_explicit_target_region_configuration_is_rejected(regions):
    with pytest.raises(ValueError, match="target_regions"):
        hybrid_output_schema(target_regions=regions)
