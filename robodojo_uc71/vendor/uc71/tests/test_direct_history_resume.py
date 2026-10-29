"""Resume proofs use a fake robot and recorded fake Responses, never the API."""

import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from gpt_field.astra import APIError
from gpt_field.direct_history import DirectHistoryPolicy, TASK
from gpt_field.direct_history_resume import load_resume, restore_policy, verify_source_hashes
from gpt_field.experiment import digest
from gpt_field.libero import LiberoConfig
from gpt_field.observation import PolicyObservation, image_to_data_url
from gpt_field.schema import GridSpec


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("direct_history_resume_fixture_runner",
                                               ROOT / "scripts/evaluate_bowl_direct_history.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def response(index):
    return {"id": f"resp_fixture_{index}", "status": "completed", "output": [
        {"id": f"rs_fixture_{index}", "type": "reasoning", "summary": [],
         "encrypted_content": f"opaque_fixture_{index}"},
        {"id": f"msg_fixture_{index}", "type": "message", "role": "assistant", "phase": "final_answer",
         "content": [{"type": "output_text", "text": json.dumps({"actions": [
             [.02, -.01, 0., 0., 0., 0., -1. if index % 2 else 1.] for _ in range(5)]})}]}],
        "usage": {"input_tokens": 100, "output_tokens": 100,
                  "input_tokens_details": {"cached_tokens": 0},
                  "output_tokens_details": {"reasoning_tokens": 30}}}


class Recorder:
    def __init__(self, path, **kwargs):
        path.write_bytes(b"OFFLINE TEST VIDEO PLACEHOLDER")

    def observe(self, *args, **kwargs):
        pass

    decision = executed = close = observe


class Simulator:
    def __init__(self, config):
        image = image_to_data_url(np.zeros((2, 2, 3), dtype=np.uint8), size=2)
        self.obs = PolicyObservation(image, image, (0, 0, 1, 0, 0, 0, .04, -.04), TASK)
        self.metadata = {"test_fixture": True}
        self.status = {"steps": 0, "success": False, "terminated": False, "horizon_reached": False}

    def start(self):
        pass

    def list_tasks(self):
        return [{"task_id": 0, "prompt": TASK}]

    def reset(self, *_):
        return self.obs

    def step(self, action):
        state = list(self.obs.state)
        for axis in range(3):
            state[axis] += action[axis] * .025
        self.obs = PolicyObservation(self.obs.image, self.obs.wrist_image, tuple(state), TASK)
        self.status["steps"] += 1
        return self.obs, False

    def close(self):
        pass


@pytest.fixture
def parent(tmp_path, monkeypatch):
    output = tmp_path / "parent"
    policies = []

    def policy_factory(client):
        policy = DirectHistoryPolicy(client)
        policies.append(policy)
        return policy

    calls = []

    def transport(*args):
        calls.append(args[1])
        assert len(calls) <= 2, "Test fixture should stop at its two-request budget"
        return response(len(calls))

    monkeypatch.setattr(runner, "DirectHistoryPolicy", policy_factory)
    summary = runner.execute_episode(output, config=runner.configuration(26.875),
        sim_config=LiberoConfig(suite="libero_spatial", seed=7, max_steps=None),
        api_key="TEST-ONLY-SECRET", sim_factory=Simulator, recorder_factory=Recorder, transport=transport)
    assert summary["status"] == "budget_exceeded" and summary["steps"] == 10
    return output, policies[0]


def edit_json(path, change):
    data = json.loads(path.read_text())
    change(data)
    path.write_text(json.dumps(data))


def test_exact_conversation_next_request_and_global_ordinal_restore(parent, tmp_path):
    source, original = parent
    expected = original.client.build_payload(original._expected_observation.to_dict(), GridSpec(), "direct")
    resume = load_resume(source, max_cost_usd=300)
    assert resume.steps == 10 and resume.api_calls == 2
    assert len(resume.observations) == len(resume.sim_statuses) == 11
    assert resume.verification["model_api_calls"] == 0
    assert resume.config.max_cost_usd == 300
    assert resume.budget["api_calls"] == 2
    transport_calls = []

    def transport(endpoint, payload, headers, timeout):
        transport_calls.append(payload)
        assert payload == expected
        return response(3)

    target = tmp_path / "child-api"
    policy = restore_policy(resume, api_key="RESTORED-TEST-SECRET", transport=transport, log_dir=target)
    assert not target.exists() and transport_calls == []
    assert policy.client.history_snapshot() == original.client.history_snapshot()
    assert policy.latest_feedback == original.latest_feedback
    assert policy._progress == original._progress
    assert policy.client.budget_snapshot()["input_tokens"] == 200
    decision = policy.act(resume.final_observation)
    assert decision.diagnostics["step"] == 10 and decision.diagnostics["chunk_index"] == 2
    assert len(transport_calls) == 1
    record = json.loads(next(target.glob("*.request.json")).read_text())
    assert record["request_ordinal"] == 3
    assert record["request_key"] == resume.next_request_key
    assert record["configuration"]["max_cost_usd"] == 300
    assert policy.client.budget_snapshot()["api_calls"] == 3
    assert policy.client.budget_snapshot()["input_tokens"] == 300
    assert "RESTORED-TEST-SECRET" not in "".join(p.read_text() for p in target.glob("*.json"))


@pytest.mark.parametrize("cap", [None, 26.875, 20, True, float("inf"), float("nan")])
def test_requires_explicit_finite_total_cap_increase(parent, cap):
    with pytest.raises(ValueError, match="finite increase"):
        load_resume(parent[0], max_cost_usd=cap)


def test_next_request_requires_room_for_full_inherited_reservation(parent):
    with pytest.raises(ValueError, match="reserve the next"):
        load_resume(parent[0], max_cost_usd=26.876)


def test_parent_identity_hash_must_match(parent):
    source, _ = parent
    edit_json(source / "manifest.json", lambda data: data["configuration"].update(task="wrong"))
    with pytest.raises(ValueError, match="identity hash"):
        load_resume(source)


def test_changed_frozen_source_digest_rejected(parent):
    source, _ = parent

    def mutate(data):
        data["configuration"]["software"]["source_sha256"]["direct_history.py"] = "0" * 64
        data["identity_hash"] = digest(data["configuration"])

    edit_json(source / "manifest.json", mutate)
    with pytest.raises(ValueError, match="source changed"):
        load_resume(source)


def test_tampered_conversation_rejected_before_any_real_transport(parent, monkeypatch):
    source, _ = parent
    path = sorted((source / "api").glob("*.request.json"))[1]
    edit_json(path, lambda data: data["payload"]["input"][1].update(encrypted_content="tampered"))
    monkeypatch.setattr("gpt_field.astra.urllib_transport", lambda *a: pytest.fail("Network forbidden"))
    with pytest.raises(APIError, match="no automatic retry"):
        load_resume(source)


def test_missing_usage_is_not_dropped(parent):
    source, _ = parent
    path = sorted((source / "api").glob("*.response.json"))[1]
    edit_json(path, lambda data: data.pop("usage"))
    with pytest.raises(APIError):
        load_resume(source)


def test_unmatched_request_is_not_retried_or_ignored(parent):
    source, _ = parent
    (source / "api/000003-pending.request.json").write_text("{}")
    with pytest.raises(ValueError, match="request count"):
        load_resume(source)


def test_missing_response_rejected(parent):
    source, _ = parent
    next((source / "api").glob("*.response.json")).unlink()
    with pytest.raises(ValueError, match="missing"):
        load_resume(source)


def test_modified_measured_feedback_rejected(parent):
    source, _ = parent
    trace_path = source / "trace.jsonl"
    rows = [json.loads(line) for line in trace_path.read_text().splitlines()]
    rows[-1]["feedback"]["measured_translation_delta_m"][0] += .001
    trace_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="measured feedback"):
        load_resume(source)


def test_pending_action_and_incomplete_trace_fail_closed(parent):
    source, _ = parent
    path = source / "trace.jsonl"
    path.write_text(path.read_text().rsplit("\n", 2)[0] + "\n")
    with pytest.raises(ValueError, match="trace sequence"):
        load_resume(source)


def test_summary_disagreement_rejected(parent):
    source, _ = parent
    edit_json(source / "summary.json", lambda data: data.update(steps=5))
    with pytest.raises(ValueError, match="summary files disagree"):
        load_resume(source)


def test_parent_artifact_changes_after_verification_block_restore(parent, tmp_path):
    source, _ = parent
    resume = load_resume(source)
    (source / "simulator.json").write_text("{}")
    with pytest.raises(ValueError, match="Parent artifact changed"):
        verify_source_hashes(resume)
    with pytest.raises(ValueError, match="Parent artifact changed"):
        restore_policy(resume, api_key="TEST", log_dir=tmp_path / "unused")
    assert not (tmp_path / "unused").exists()


def test_nonempty_destination_and_modified_history_block_restore(parent, tmp_path):
    resume = load_resume(parent[0])
    destination = tmp_path / "api"
    destination.mkdir()
    (destination / "prior.request.json").write_text("{}")
    with pytest.raises(ValueError, match="not empty"):
        restore_policy(resume, api_key="TEST", log_dir=destination)
    resume._policy.client._history[1]["encrypted_content"] = "mutated"
    with pytest.raises(ValueError, match="conversation was modified"):
        restore_policy(resume, api_key="TEST")


def test_restore_preserves_all_budget_fields_except_explicit_cap(parent):
    resume = load_resume(parent[0], max_cost_usd=300)
    previous = copy.deepcopy(resume.summary["budget"])
    updated = copy.deepcopy(resume.budget)
    updated["max_cost_usd"] = previous["max_cost_usd"]
    updated["configuration"]["max_cost_usd"] = previous["configuration"]["max_cost_usd"]
    assert updated == previous
    policy = restore_policy(resume, api_key="TEST")
    assert policy.client.config.max_api_calls is None and policy.client.config.max_wall_time_s is None
    assert policy.client.budget_snapshot()["unknown_usage_calls"] == 0
    assert policy.client.budget_snapshot()["cost_usd_upper_bound"] == previous["cost_usd_upper_bound"]


def test_in_memory_ledger_mutation_cannot_drop_inherited_cost(parent):
    resume = load_resume(parent[0])
    resume.budget["cost_usd_upper_bound"] = 0
    with pytest.raises(ValueError, match="inherited budget was modified"):
        restore_policy(resume, api_key="TEST")


def test_in_memory_rolling_feedback_mutation_cannot_change_policy(parent):
    resume = load_resume(parent[0])
    resume._policy._progress.clear()
    with pytest.raises(ValueError, match="feedback state was modified"):
        restore_policy(resume, api_key="TEST")


def test_new_parent_request_after_verification_cannot_escape_ledger(parent):
    source, _ = parent
    resume = load_resume(source)
    (source / "api/000003-new.request.json").write_text("{}")
    with pytest.raises(ValueError, match="API record set changed"):
        restore_policy(resume, api_key="TEST")
