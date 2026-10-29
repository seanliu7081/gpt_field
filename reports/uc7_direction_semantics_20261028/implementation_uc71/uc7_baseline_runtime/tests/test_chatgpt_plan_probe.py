"""Subscription access probe checks use simulated network responses only."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from gpt_field.astra import APIError
from gpt_field.observation import PolicyObservation, image_to_data_url


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("chatgpt_plan_field_probe",
    ROOT / "scripts/chatgpt_plan_field_probe.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


@pytest.fixture
def trace(tmp_path):
    image = image_to_data_url(np.zeros((224, 224, 3), dtype=np.uint8), rotate=False)
    obs = PolicyObservation(image, image, (0, 0, 1, 0, 0, 0, 0.04, -0.04), probe.TASK)
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text(json.dumps({"event": "reset", "observation": obs.to_dict(),
        "sim_status": {"success": False, "oracle_marker": "DO_NOT_SEND"}}) + "\n")
    return trace_path


def test_real_client_preserves_prompt_and_planner_without_execution(tmp_path, trace, monkeypatch):
    from gpt_field.chatgpt_plan import ChatGPTPlanFieldClient
    from gpt_field.schema import GridSpec
    monkeypatch.setenv("OPENAI_API_KEY", "UNAUTHORIZED_API_KEY")
    calls = []
    token = "SUBSCRIPTION_TEST_CREDENTIAL"

    def send(endpoint, payload, headers, timeout_s):
        calls.append(payload)
        assert headers["Authorization"] == "Bearer " + token
        assert payload["model"] == "gpt-6-astra"
        assert payload["stream"] and not payload["store"]
        assert "max_output_tokens" not in payload
        assert "DO_NOT_SEND" not in json.dumps(payload)
        grid = GridSpec()
        field = {"stage": "mock_field_for_transport_test", "motion_mode": "move",
                 "gripper_command": "open", "orientation_target_xyzw": [0, 0, 0, 1],
                 "values": (-grid.coordinates((0, 0, 0))[:, 0]).tolist(),
                 "blocked_indices": [], "unknown_indices": []}
        return {"id": "mock_response", "model": "gpt-6-astra", "status": "completed",
                "output": [{"type": "message", "role": "assistant", "content": [
                    {"type": "output_text", "text": json.dumps(field)}]}],
                "usage": {"input_tokens": 100, "output_tokens": 100}}

    def factory(config, **kwargs):
        return ChatGPTPlanFieldClient(config, transport=send, **kwargs)

    output = tmp_path / "probe"
    result = probe.probe(output, lambda: token, trace_path=trace,
        model_catalog=lambda _: [{"slug": "gpt-6-astra"}], client_factory=factory)
    assert result["astra_subscription_access_verified"]
    assert result["validated_field_generations"] == result["local_plans"] == len(calls) == 1
    assert result["simulator_actions"] == 0 and result["success_rate"] is None
    assert result["api_key_calls"] == 0
    for path in output.rglob("*.json"):
        assert token not in path.read_text()
        assert "UNAUTHORIZED_API_KEY" not in path.read_text()
    plan = json.loads((output / "plan.json").read_text())
    assert len(plan["action_not_executed"]) == 7
    assert "plan" in plan["diagnostics"]


@pytest.mark.parametrize("catalog", [[], [{"slug": "gpt-6.1-sol"}], [{"slug": "gpt-6-astra-other"}]])
def test_missing_exact_model_never_substitutes_or_infers(tmp_path, trace, catalog):
    result = probe.probe(tmp_path / "probe", lambda: pytest.fail("Must not infer"),
        trace_path=trace, model_catalog=lambda _: catalog)
    assert result["status"] == "astra_not_in_account_catalog"
    assert result["validated_field_generations"] == 0
    assert not result["astra_subscription_access_verified"]
    assert result["usage"]["api_calls"] == 0


def test_auth_exception_does_not_leak_or_fallback(tmp_path, trace, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "SHOULD_NOT_BE_USED")

    def denied(_):
        raise APIError("secret-access-token-in-remote-error")

    output = tmp_path / "probe"
    result = probe.probe(output, lambda: "", trace_path=trace, model_catalog=denied)
    assert result["status"] == "subscription_probe_failed"
    assert result["usage"]["api_calls"] == 0
    assert "secret-access-token" not in (output / "summary.json").read_text()


def test_completed_astra_response_proves_access_even_if_field_is_invalid(tmp_path, trace):
    from gpt_field.chatgpt_plan import ChatGPTPlanFieldClient

    def factory(config, **kwargs):
        return ChatGPTPlanFieldClient(config, transport=lambda *args: {
            "id": "mock-completed-invalid-field", "status": "completed",
            "model": probe.MODEL, "output": [],
            "usage": {"input_tokens": 100, "output_tokens": 20}}, **kwargs)

    result = probe.probe(tmp_path / "probe", lambda: "fake-oauth-test-token",
        trace_path=trace, model_catalog=lambda _: [{"slug": probe.MODEL}],
        client_factory=factory)
    assert result["status"] == "subscription_probe_failed"
    assert result["astra_subscription_access_verified"]
    assert not result["field_pipeline_verified"]
    assert result["validated_field_generations"] == result["local_plans"] == 0
    assert result["simulator_actions"] == 0


def test_preserves_existing_run_before_any_network(tmp_path, trace):
    output = tmp_path / "existing"
    output.mkdir()
    (output / "keep.txt").write_text("original")
    with pytest.raises(FileExistsError):
        probe.probe(output, lambda: pytest.fail("Must not authenticate"), trace_path=trace)
    assert (output / "keep.txt").read_text() == "original"


def test_source_must_be_initial_observation(tmp_path, trace):
    original = json.loads(trace.read_text())
    original["event"] = "executed"
    trace.write_text(json.dumps(original))
    with pytest.raises(ValueError, match="reset"):
        probe.initial_observation(trace)


def test_oracle_fields_rejected_inside_observation(trace):
    original = json.loads(trace.read_text())
    original["observation"]["reward"] = 0
    trace.write_text(json.dumps(original))
    with pytest.raises(ValueError, match="allowlisted"):
        probe.initial_observation(trace)
