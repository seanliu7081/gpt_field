"""Opt-in attempt context only: offline payload, transaction and execution checks."""
import copy
from dataclasses import replace
import json

import pytest

from gpt_field.astra import APIError, SchemaError
from gpt_field.attempt_memory import ATTEMPT_MEMORY_VERSION, attempt_memory_instructions, attempt_review_schema
from gpt_field.speed_first_v3 import (
    ChatGPTPlanSpeedFirstV3Client, FINE_GRID, speed_first_v3_instructions, speed_first_v3_schema,
)
from test_field_history import FakeTransport, envelope
from test_field_history_runner import FakeRecorder, observation, rows
from test_hybrid_subscription_runner import runner, sim_config
from test_speed_first_runner import monitored_simulator
from test_speed_first_v3_runner import OPTIONS, highres_observation, v3_intent
from test_target_identity_runner import DESCRIPTION


def review(previous=None, *, next_change="Inspect the current object and choose a visible approach."):
    return {"previous_attempt": previous, "next_change": next_change, "prerequisites": []}


def with_review(raw, previous=None, **kwargs):
    return {**copy.deepcopy(raw), "interaction_review": review(previous, **kwargs)}


def memory_items(messages):
    result = []
    for message in messages:
        for part in message.get("content", []):
            if part.get("type") not in ("input_text", "output_text"):
                continue
            try:
                value = json.loads(part["text"])
            except (KeyError, TypeError, ValueError):
                continue
            if isinstance(value, dict) and value.get("type") == "attempt_memory":
                result.append(value)
    return result


@pytest.mark.parametrize("task", ["bowl_plate", "drawer_bowl"])
def test_enabled_payload_adds_only_prompt_review_schema_and_current_memory(task, highres_observation):
    kwargs = dict(task_key=task, robot_geometry_guidance=True, target_description=DESCRIPTION,
                  token_provider=lambda: pytest.fail("credentials requested"))
    baseline = ChatGPTPlanSpeedFirstV3Client(**kwargs)
    explicit_off = ChatGPTPlanSpeedFirstV3Client(attempt_memory=False, **kwargs)
    enabled = ChatGPTPlanSpeedFirstV3Client(attempt_memory=True, **kwargs)
    highres_observation = replace(highres_observation, prompt=baseline.task_prompt)
    original = baseline.build_payload(highres_observation.to_dict())
    assert explicit_off.build_payload(highres_observation.to_dict()) == original
    assert explicit_off.history_stats() == baseline.history_stats()
    changed = enabled.build_payload(highres_observation.to_dict())
    assert changed["instructions"] == original["instructions"] + "\n\n" + attempt_memory_instructions()
    assert enabled.prompt_version == baseline.prompt_version + "+" + ATTEMPT_MEMORY_VERSION
    assert len(memory_items(changed["input"])) == 1
    assert memory_items(changed["input"])[0]["attempts"] == []
    assert memory_items(enabled.history_snapshot()) == []
    changed["instructions"] = original["instructions"]
    changed["input"].pop()
    schema = changed["text"]["format"]["schema"]
    assert schema["properties"].pop("interaction_review") == attempt_review_schema()
    schema["required"].remove("interaction_review")
    assert changed == original


@pytest.mark.parametrize("flags", [[], ["--speed-first"], ["--speed-first-v2"],
    ["--distance-field-control"], ["--controller-mode", "direct"],
    ["--speed-first-v3", "--controller-mode", "direct"]])
def test_cli_rejects_memory_without_v3_before_side_effects(flags, monkeypatch):
    monkeypatch.setattr(runner, "configuration", lambda: pytest.fail("invalid args passed validation"))
    with pytest.raises(SystemExit) as caught:
        runner.main(["--attempt-memory", "--preflight", *flags])
    assert caught.value.code == 2


@pytest.mark.parametrize("value", [None, 0, 1, "true", [], {}])
def test_memory_boolean_contract(value, tmp_path, monkeypatch):
    for method in (ChatGPTPlanSpeedFirstV3Client, speed_first_v3_instructions, speed_first_v3_schema):
        with pytest.raises(ValueError, match="attempt_memory"):
            method(attempt_memory=value)
    monkeypatch.setattr(runner, "_validate_collision_configuration", lambda *a, **k: pytest.fail("side effect"))
    with pytest.raises(ValueError, match="Attempt memory"):
        runner.execute_episode(tmp_path / "absent", config=runner.configuration(),
            sim_config=sim_config(), attempt_memory=value, **OPTIONS)
    assert not (tmp_path / "absent").exists()


def test_preflight_records_provenance_without_starting_task(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(runner, "SimProcess", lambda *a: pytest.fail("simulator called"))
    monkeypatch.setattr(ChatGPTPlanSpeedFirstV3Client, "predict", lambda *a, **k: pytest.fail("model called"))
    output = tmp_path / "not_created"
    assert runner.main(["--speed-first-v3", "--attempt-memory", "--task", "drawer_bowl",
        "--robot-geometry-guidance", "--target-description", DESCRIPTION,
        "--preflight", "--output", str(output)]) == 0
    data = json.loads(capsys.readouterr().out)
    metadata = runner._attempt_memory_metadata(True)["attempt_memory"]
    assert data["attempt_memory"] == data["policy"]["attempt_memory"] == metadata
    assert metadata["version"] == ATTEMPT_MEMORY_VERSION
    assert not metadata["model_claims_independently_verified"]
    assert metadata["additional_inference_calls"] == 0 and not metadata["execution_gates_added"]
    assert data["api_calls"] == data["api_key_calls"] == 0
    assert data["stopping_rule"][0] == "LIBERO success" and not data["stop_on_model_complete"]
    assert data["policy"]["direct_action_steps"] == 5
    assert data["policy"]["field_update_steps"] == 30
    assert data["simulator"]["stop_on_collision"] is False
    assert data["max_wall_time_s"] is None
    assert not output.exists()


def test_review_validation_atomic_and_no_speculative_or_failed_history(highres_observation):
    raw = with_review(v3_intent(direct=True, phase="grasp"))
    bad_review = copy.deepcopy(raw)
    bad_review["interaction_review"]["extra"] = "MALFORMED_UNCOMMITTED_REVIEW"
    bad_action = copy.deepcopy(raw)
    bad_action["direct"]["actions"][0][0] = 4.
    fake = FakeTransport(envelope(raw), envelope(bad_review), envelope(bad_action), APIError("offline failure"))
    client = ChatGPTPlanSpeedFirstV3Client(attempt_memory=True,
        token_provider=lambda: "offline-token", transport=fake)
    first = client.predict(highres_observation.to_dict())
    assert "interaction_review" not in first
    assert client.last_compact_prediction == raw
    before = client.history_snapshot()
    observation_count = client._observation_count
    accepted_review = copy.deepcopy(client.last_interaction_review)
    for invalid in (bad_review, bad_action):
        with pytest.raises(SchemaError):
            client.validate_response(invalid, FINE_GRID)
        assert client.last_interaction_review == accepted_review
        assert client.last_compact_prediction == raw
        assert client.history_snapshot() == before
    for error in (SchemaError, SchemaError, APIError):
        with pytest.raises(error):
            client.predict(highres_observation.to_dict())
        assert client.history_snapshot() == before
        assert client._observation_count == observation_count
        # Inherited predict clears executable diagnostics on failure; never reuse a stale action.
        assert client.last_interaction_review is None
        assert client.last_compact_prediction is None
        assert memory_items(client.history_snapshot()) == []
    assert "MALFORMED_UNCOMMITTED_REVIEW" not in json.dumps(client.build_payload(highres_observation.to_dict()))
    # Payload previews do not accept their observation or create attempts.
    preview = client.build_payload(highres_observation.to_dict())
    assert client.build_payload(highres_observation.to_dict()) == preview
    assert client.history_snapshot() == before
    client.reset_history()
    assert client.history_snapshot() == [] and client._observation_count == 0
    assert client.last_attempt_memory is None and client.last_interaction_review is None
    assert memory_items(client.build_payload(highres_observation.to_dict())["input"])[0]["attempts"] == []


def test_only_one_current_memory_even_after_history_bounding(highres_observation):
    raw = with_review(v3_intent(direct=True), next_change="Continue from CURRENT RGB, never replay a past command.")
    fake = FakeTransport(*(envelope(raw, id=f"offline-{i}") for i in range(14)))
    client = ChatGPTPlanSpeedFirstV3Client(attempt_memory=True,
        token_provider=lambda: "offline-token", transport=fake)
    for _ in range(14):
        client.predict(highres_observation.to_dict())
        assert memory_items(client.history_snapshot()) == []
    payload = client.build_payload(highres_observation.to_dict())
    assert client.last_history_presentation["older_observations"] > 0
    memory = memory_items(payload["input"])
    assert len(memory) == 1 and len(memory[0]["attempts"]) <= 3
    assert len(json.dumps(memory[0])) < 12000
    assert all(len(memory_items(call["input"])) == 1 for call in fake.calls)


def test_same_gripper_direct_and_field_actions_with_review_memory(tmp_path, highres_observation):
    direct = v3_intent(direct=True, phase="verify")
    direct["target_region"] = "black_bowl"
    direct["direct"]["actions"] = [[.01 * (i + 1), 0., 0., 0., 0., 0., 1.] for i in range(5)]
    recover = v3_intent(phase="recover", goal=(0., -.03, 0.))
    recover["target_region"] = "black_bowl"
    raw = [v3_intent(operation="close", phase="grasp"), direct, recover]
    assertion = {"attempt_id": "attempt-0001", "outcome": "not_engaged",
                 "evidence": "OFFLINE_MODEL_CLAIM: object appears separated after measured movement."}
    results, executed, requests = [], [], []
    for enabled in (False, True):
        output = tmp_path / str(enabled)
        base, instances = monitored_simulator(highres_observation, success_at=21, collision_at=1)
        class Simulator(base):
            def step(self, action):
                measured, success = super().step(action)
                if action[6] != 0:
                    state = list(measured.state)
                    target = .016 if action[6] > 0 else .04
                    magnitude = abs(state[6])
                    magnitude += max(-.004, min(.004, target - magnitude))
                    state[6:8] = [magnitude, -magnitude]
                    self.obs = replace(measured, state=tuple(state))
                return self.obs, success
        responses = [with_review(value, assertion if i else None,
            next_change="Move the grasp laterally using the current image.") if enabled else value
            for i, value in enumerate(raw)]
        fake = FakeTransport(*(envelope(value, id=f"offline-{i}", model="gpt-6-astra",
            reasoning={"effort": "medium"}) for i, value in enumerate(responses)))
        result = runner.execute_episode(output, config=runner.configuration(),
            sim_config=sim_config(render_size=512, image_size=512,
                collision_check_mode="task_manipulation_v1", stop_on_collision=False),
            token_provider=lambda: "offline-token", transport=fake,
            sim_factory=Simulator, recorder_factory=FakeRecorder, attempt_memory=enabled, **OPTIONS)
        assert result["status"] == "success", result.get("error")
        assert result["steps"] == result["measured_feedback_count"] == 21
        assert result["local_gripper_actions"] == 15 and result["ordinary_direct_actions"] == 5
        assert result["coarse_field_actions"] == 1 and len(fake.calls) == 3
        assert instances[0].actions[15:20] == direct["direct"]["actions"]
        assert result["libero_success"] and not result["model_declared_complete"]
        assert result["collision"]["detected"] and result["failure_reason"] is None
        assert result["api_key_calls"] == 0 and instances[0].closed
        manifest = json.loads((output / "manifest.json").read_text())["configuration"]
        state = json.loads((output / "run_state.json").read_text())
        for document in (manifest, manifest["policy"], state, result):
            if enabled:
                assert document["attempt_memory"] == runner._attempt_memory_metadata(True)["attempt_memory"]
            else:
                assert "attempt_memory" not in document
        results.append(result)
        executed.append(instances[0].actions)
        requests.append(fake.calls)
    assert executed[0] == executed[1]
    for call in requests[1]:
        assert len(memory_items(call["input"])) == 1
        assert "CONTACT_MUST_NOT_ENTER_POLICY" not in json.dumps(call)
    # Review from the prior accepted response persists in the next memory, with attribution.
    final_memory = memory_items(requests[1][2]["input"])[0]
    assert "OFFLINE_MODEL_CLAIM" in json.dumps(final_memory)
    assert len(final_memory["attempts"]) <= 3
