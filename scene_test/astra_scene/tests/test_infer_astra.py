"""Offline transport boundary tests: never contact authentication or inference."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import pytest

SOURCE = Path(__file__).resolve().parents[1] / "infer_astra.py"
spec = importlib.util.spec_from_file_location("astra_scene_infer_test_module", SOURCE)
infer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(infer)


def _schema():
    return {"type": "object", "properties": {"field": {"type": "number"}}, "required": ["field"], "additionalProperties": False}


def _validate(value):
    if not isinstance(value, dict) or set(value) != {"field"}:
        raise ValueError("Invalid test field")
    return {**value, "_validation": {"normalizations": []}}


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    inputs = tmp_path / "rgb"
    inputs.mkdir()
    entries = []
    for index, name in enumerate(infer.EXPECTED_NAMES):
        path = inputs / f"{name}.png"
        Image.new("RGB", (512, 512), (index * 40, 70, 90)).save(path)
        entries.append({"name": name, "rgb": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    (inputs / "rgb_inputs.json").write_text(json.dumps({"schema_version": 1, "inputs": entries}))
    monkeypatch.setattr(infer, "_field_module", lambda: SimpleNamespace(field_schema=_schema, validate_field=_validate))
    monkeypatch.setattr(infer, "_runtime_sources", lambda: [SOURCE])
    def no_auth(*args, **kwargs):
        raise AssertionError("Preparation accessed credentials")
    monkeypatch.setattr(infer, "SubscriptionCredentials", no_auth)
    output = tmp_path / "prepared"
    manifest = infer.prepare(inputs / "rgb_inputs.json", output)
    return output, manifest


def _response(text='{"field": 1.0}', *, model=infer.MODEL, effort="medium"):
    return {"status": "completed", "model": model, "reasoning": {"effort": effort},
            "output": [{"type": "message", "content": [{"type": "output_text", "text": text}]}],
            "usage": {"input_tokens": 10, "output_tokens": 20}}


def test_prepare_is_network_and_credential_free_and_exact_pixels(prepared):
    root, manifest = prepared
    assert manifest["status"] == "prepared"
    assert manifest["subscription_inference_calls"] == manifest["api_key_calls"] == 0
    assert manifest["credentials_accessed"] is False
    assert len(manifest["requests"]) == 3
    for item, request in zip(manifest["inputs"], manifest["requests"]):
        payload = json.loads((root / request["file"]).read_text())
        assert payload == infer.build_payload((root / item["file"]).read_bytes(), _schema())
        assert payload["model"] == infer.MODEL and payload["reasoning"] == {"effort": "medium"}
        assert payload["store"] is False and payload["stream"] is True
        assert len(payload["input"]) == 1
        assert not any(name in payload["instructions"] for name in infer.EXPECTED_NAMES)
        assert "DepthPro" not in payload["instructions"] and "LIBERO" not in payload["instructions"]
        assert (root / request["wire_file"]).read_text() == infer.ordered_dumps(payload)


def test_three_independent_calls_and_saved_provenance(prepared, tmp_path):
    root, plan = prepared
    calls = []
    def send(endpoint, payload, headers, timeout):
        calls.append(copy.deepcopy(payload))
        assert endpoint == infer.ENDPOINT and timeout == 300
        assert headers["Authorization"] == "Bearer oauth-offline-fixture"
        return _response()
    result = infer.run_prepared(root, tmp_path / "run", token_provider=lambda: "oauth-offline-fixture", transport=send)
    assert result["status"] == "complete"
    assert result["inference_attempts"] == len(calls) == 3
    assert result["billing"] == "offline_mock" and result["api_key_calls"] == 0
    assert infer._all_integrity(result["integrity"])
    assert all(len(payload["input"]) == 1 for payload in calls)
    assert len({p["input"][0]["content"][1]["image_url"] for p in calls}) == 3
    for record in result["predictions"]:
        assert record["status"] == "valid" and record["usage"]["output_tokens"] == 20
        assert record["response_model"] == infer.MODEL and record["response_reasoning_effort"] == "medium"
        for key in ("field", "raw_field", "raw_response"):
            assert infer.sha256(tmp_path / "run" / record[key]["file"]) == record[key]["sha256"]
    assert not any("oauth-offline-fixture" in path.read_text() for path in (tmp_path / "run").rglob("*.json"))


@pytest.mark.parametrize("response", [
    _response(model="different-model"), _response(effort="high"),
    _response('{"wrong":1}'), _response('{"field":1,"field":2}'),
    _response('{"field":NaN}'), {**_response(), "status": "incomplete"},
])
def test_bad_model_effort_schema_or_completion_stops_without_retry(prepared, tmp_path, response):
    root, _ = prepared
    calls = []
    def send(*args):
        calls.append(1)
        return response
    result = infer.run_prepared(root, tmp_path / "run", token_provider=lambda: "oauth-fixture", transport=send)
    assert result["status"] == "failed" and len(calls) == 1
    assert result["inference_attempts"] == 1 and len(result["predictions"]) == 1
    assert result["predictions"][0]["status"] == "failed"


def test_arbitrary_failure_details_and_credentials_are_not_logged(prepared, tmp_path):
    root, _ = prepared
    def send(*args):
        raise RuntimeError("secret oauth-fixture-private-value HTTP body")
    result = infer.run_prepared(root, tmp_path / "run", token_provider=lambda: "oauth-fixture-private-value", transport=send)
    assert result["status"] == "failed" and result["error_type"] == "RuntimeError"
    text = (tmp_path / "run/manifest.json").read_text()
    assert "oauth-fixture-private-value" not in text and "HTTP body" not in text


def test_tampered_preparation_fails_before_auth_or_output(prepared, tmp_path):
    root, plan = prepared
    (root / plan["requests"][0]["file"]).write_text("{}")
    def forbidden():
        raise AssertionError("Must not access credentials")
    with pytest.raises(ValueError):
        infer.run_prepared(root, tmp_path / "run", token_provider=forbidden)
    assert not (tmp_path / "run").exists()


def test_existing_output_is_never_reused(prepared, tmp_path):
    root, _ = prepared
    existing = tmp_path / "existing"
    existing.mkdir()
    (existing / "keep").write_text("preserve")
    with pytest.raises(FileExistsError):
        infer.run_prepared(root, existing)
    assert (existing / "keep").read_text() == "preserve"


def test_api_key_provider_value_rejected_without_inference(prepared, tmp_path):
    root, _ = prepared
    def forbidden(*args):
        raise AssertionError("API key must not reach inference")
    result = infer.run_prepared(root, tmp_path / "run", token_provider=lambda: "sk-disallowed", transport=forbidden)
    assert result["status"] == "failed" and result["inference_attempts"] == 0
    assert result["error_type"] == "MissingCredentials"
