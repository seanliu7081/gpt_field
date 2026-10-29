"""Bowl/plate demonstration conditioning preserves existing drawer protocols."""

import copy
import hashlib
import json
from unittest.mock import Mock

import pytest

from gpt_field.astra import SchemaError
from gpt_field.demo_direct_subscription import (
    ChatGPTPlanDemoDirectClient, DEMO_DIRECT_PROMPT_VERSION, DEMO_DIRECT_REPRESENTATION,
    DEMO_DIRECT_SCHEMA_NAME, demo_direct_instructions, demo_direct_output_schema,
    demo_direct_prompt_version, demo_direct_representation, demo_direct_schema_name,
    shared_demo_task_guidance,
)
from gpt_field.demonstration import load_demonstration
from gpt_field.distance_subscription import (
    ChatGPTPlanDistanceClient, DISTANCE_PROMPT_VERSION, distance_instructions,
    distance_output_schema, distance_prompt_version, distance_schema_name,
)
from gpt_field.distance_tasks import resolve_distance_task
from test_block_feedback_history import feedback, text_records
from test_demo_direct_subscription import prediction as direct_prediction, response
from test_demonstration import make_bundle, write_bundle
from test_distance_subscription import prediction as hybrid_prediction
from test_field_history import FakeTransport, observation


BOWL = resolve_distance_task("bowl_plate")


@pytest.fixture
def bowl_demo(tmp_path):
    bundle = make_bundle()
    bundle.update(task_key=BOWL.key, task_prompt=BOWL.prompt)
    bundle["source"].update(suite=BOWL.suite, task_id=BOWL.task_id,
                            file_name="pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo.hdf5")
    return load_demonstration(write_bundle(tmp_path, bundle), task_key=BOWL.key)


def test_all_historical_default_protocols_remain_byte_identical():
    before = {"bowl": distance_instructions(),
              "drawer": distance_instructions(task_key="drawer_bowl"),
              "drawer_demo": distance_instructions(task_key="drawer_bowl", demonstration=True),
              "direct": demo_direct_instructions(), "schema": demo_direct_output_schema()}
    assert hashlib.sha256(json.dumps(before, sort_keys=True).encode()).hexdigest() == (
        "4b9b8107a51508d22656e909a32dd53ad2f0795806928d052493fbd169688988")
    assert demo_direct_prompt_version() == DEMO_DIRECT_PROMPT_VERSION
    assert demo_direct_schema_name() == DEMO_DIRECT_SCHEMA_NAME
    assert demo_direct_representation() == DEMO_DIRECT_REPRESENTATION
    assert distance_prompt_version() == DISTANCE_PROMPT_VERSION


def test_bowl_demo_versions_schemas_and_guidance_name_only_the_requested_task():
    hybrid = distance_instructions(task_key=BOWL.key, demonstration=True)
    direct = demo_direct_instructions(task_key=BOWL.key)
    assert distance_prompt_version(BOWL.key, demonstration=True) == "astra-distance-field-local-feedback-bowl-plate-one-demo-v1"
    assert demo_direct_prompt_version(BOWL.key) == "astra-direct-bowl-plate-one-demo-v1"
    assert demo_direct_schema_name(BOWL.key) == "astra_direct_bowl_plate_one_demo_v1"
    assert demo_direct_representation(BOWL.key) == "direct-actions-bowl-plate-one-demo-v1"
    for prompt in (hybrid, direct):
        assert BOWL.prompt in prompt and "external_demonstration" in prompt
        assert "bowl grasp and test lift" in prompt and "centered placement on the plate" in prompt
        assert "AFTER its labeled action_index" in prompt
        assert "demonstrations or future images are supplied" not in prompt
        assert "drawer" not in prompt.lower() and "handle" not in prompt.lower()
        assert "No reference action is executed automatically." in prompt
        assert "only the external evaluator can stop on actual LIBERO success" in prompt
    assert "bounded +Z reobservation move" in hybrid
    assert "bounded +Z reobservation move" not in direct
    assert "No distance field, local motion planner, local visual servo" in direct
    shared = shared_demo_task_guidance(task_key=BOWL.key)
    for name, text in shared.items():
        assert text in hybrid and text in direct, name
    assert "Gripper closure may require multiple five-action chunks" in shared["grasp_settling"]
    assert "Visual overlap between the bowl and plate is insufficient" in shared["placement"]
    assert "Direct five-action decision" in shared["reobservation"]
    direct_schema = demo_direct_output_schema(task_key=BOWL.key)
    for schema in (direct_schema, distance_output_schema(task_key=BOWL.key)):
        assert schema["properties"]["target_region"]["enum"] == list(BOWL.target_regions)
        assert schema["properties"]["control_context"]["properties"]["phase"]["enum"] == list(BOWL.control_phases)
    assert set(direct_schema["properties"]) == {"actions", "target_region", "routing_reason", "control_context"}


@pytest.mark.parametrize("client_type", [ChatGPTPlanDistanceClient, ChatGPTPlanDemoDirectClient])
def test_bowl_demo_preview_is_task_matched_subscription_medium_without_auth(bowl_demo, client_type):
    provider, transport = Mock(), FakeTransport()
    client = client_type(task_key=BOWL.key, demonstration=bowl_demo, token_provider=provider, transport=transport)
    payload = client.build_payload(observation().to_dict())
    assert payload["input"][0] == bowl_demo.message()
    assert payload["model"] == "gpt-6-astra" and payload["reasoning"] == {"effort": "medium"}
    assert client.task_spec == BOWL and client.task_prompt == BOWL.prompt
    assert client.target_regions == BOWL.target_regions
    assert client.chunk_size == 5 and client.image_history == 2 and client.feedback_history == "block_summary"
    assert not client.stop_on_model_complete
    assert payload["text"]["format"]["name"] == (
        distance_schema_name(BOWL.key) if client_type is ChatGPTPlanDistanceClient else demo_direct_schema_name(BOWL.key))
    assert json.loads(payload["input"][1]["content"][0]["text"])["prompt"] == BOWL.prompt
    assert not transport.calls and not client.history_snapshot()
    provider.assert_not_called()


@pytest.mark.parametrize("direct", [False, True])
def test_same_bowl_reference_is_retained_separately_across_four_model_requests(bowl_demo, direct):
    raw = (direct_prediction(phase="align", target="plate") if direct else
           hybrid_prediction(contact=True, phase="align", visual=False))
    raw["target_region"] = "plate"
    replies = [response(raw, id=f"bowl-demo-{i}") for i in range(4)]
    transport = FakeTransport(*replies)
    kind = ChatGPTPlanDemoDirectClient if direct else ChatGPTPlanDistanceClient
    client = kind(task_key=BOWL.key, demonstration=bowl_demo,
                  token_provider=lambda: "fixture-oauth", transport=transport)
    for i in range(4):
        result = client.predict(observation(i).to_dict())
        expected = raw["actions"] if direct else raw["direct"]["actions"]
        assert result["controller"] == "contact_direct" and result["direct"]["actions"] == expected
        assert client.last_compact_prediction == raw
    for i, payload in enumerate(transport.calls):
        assert payload["input"][0] == bowl_demo.message()
        assert payload["input"].count(bowl_demo.message()) == 1
        images = [p for m in payload["input"][1:] for p in m.get("content", []) if p.get("type") == "input_image"]
        assert len(images) == 2*min(i+1, 2)
    assert bowl_demo.message() not in client.history_snapshot()
    assert client.history_stats()["observations"] == 4
    assert client.history_stats()["feedback_records"] == 0
    assert client.history_stats()["retained_image_pairs"] == 2


def test_bowl_direct_feedback_compacts_at_five_actual_actions_without_field_events(bowl_demo):
    raw = direct_prediction(phase="grasp", target="black_bowl")
    transport = FakeTransport(response(raw), response(raw))
    client = ChatGPTPlanDemoDirectClient(task_key=BOWL.key, demonstration=bowl_demo,
        token_provider=lambda: "fixture-oauth", transport=transport)
    client.predict(observation().to_dict())
    for i in range(5):
        client.record_transition(feedback(i))
    client.predict(observation(1).to_dict())
    records = text_records(transport.calls[-1]["input"][1:])
    assert len(records) == 1 and records[0]["type"] == "execution_block_summary"
    assert records[0]["executed_actions"] == 5
    assert client.last_field_expansion_s == 0. and not client.last_task_complete


@pytest.mark.parametrize("phase,target", [
    ("approach", "black_bowl"), ("grasp", "black_bowl"), ("transport", "plate"),
    ("align", "plate"), ("release", "plate"), ("verify", "plate"), ("recover", "black_bowl"),
])
def test_bowl_direct_phase_routes_preserve_all_five_original_actions(bowl_demo, phase, target):
    raw = direct_prediction(phase=phase, target=target)
    client = ChatGPTPlanDemoDirectClient(task_key=BOWL.key, demonstration=bowl_demo,
        token_provider=lambda: "fixture-oauth", transport=FakeTransport(response(raw)))
    assert client.predict(observation().to_dict())["direct"]["actions"] == raw["actions"]


@pytest.mark.parametrize("change", [
    lambda raw: raw.update(target_region="drawer_interior"),
    lambda raw: raw["control_context"].update(phase="drawer_open"),
    lambda raw: raw["control_context"].update(phase="handle_grasp"),
    lambda raw: raw["control_context"].update(phase="handle_release"),
    lambda raw: raw.update(task_complete=True),
])
@pytest.mark.parametrize("direct", [False, True])
def test_bowl_clients_reject_drawer_routes_or_completion_without_committing(bowl_demo, change, direct):
    raw = direct_prediction(phase="align", target="plate") if direct else hybrid_prediction(contact=True, visual=False)
    change(raw)
    kind = ChatGPTPlanDemoDirectClient if direct else ChatGPTPlanDistanceClient
    client = kind(task_key=BOWL.key, demonstration=bowl_demo,
                  token_provider=lambda: "fixture-oauth", transport=FakeTransport(response(raw)))
    with pytest.raises(SchemaError):
        client.predict(observation().to_dict())
    assert client.history_snapshot() == [] and client.last_control_context is None


@pytest.mark.parametrize("client_type", [ChatGPTPlanDistanceClient, ChatGPTPlanDemoDirectClient])
def test_demo_from_other_task_cannot_enter_either_client(bowl_demo, client_type):
    provider = Mock()
    with pytest.raises(ValueError, match="match the selected task"):
        client_type(task_key="drawer_bowl", demonstration=bowl_demo, token_provider=provider)
    provider.assert_not_called()


def test_direct_bowl_release_and_grasp_cannot_swap_targets(bowl_demo):
    client = ChatGPTPlanDemoDirectClient(task_key=BOWL.key, demonstration=bowl_demo,
                                        token_provider=lambda: "fixture-oauth")
    for phase, target in (("release", "black_bowl"), ("grasp", "plate")):
        with pytest.raises(SchemaError):
            client.validate_response(direct_prediction(phase=phase, target=target), client.grid)
        assert client.last_control_context is None
