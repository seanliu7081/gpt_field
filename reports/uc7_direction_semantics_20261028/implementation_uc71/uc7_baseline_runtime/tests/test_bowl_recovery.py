import copy
import importlib.util
import json
from pathlib import Path

import pytest

from gpt_field.astra import APIError, AstraClient
from gpt_field.observation import PolicyObservation


ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


recovery = load("recover_bowl_continuation", ROOT / "scripts/recover_bowl_continuation.py")
fixtures = load("continuation_offline_fixtures", ROOT / "tests/test_bowl_continuation.py")


def make_chain(tmp_path):
    root, child = tmp_path / "root", tmp_path / "child"
    rows, _ = fixtures.make_source(root)
    # Real PolicyObservation records always serialize the measured state as floats.
    for row in rows:
        if "observation" in row:
            row["observation"]["observation/state"] = [float(x) for x in row["observation"]["observation/state"]]
        for key in ("state", "next_state"):
            if key in row:
                row[key] = [float(x) for x in row[key]]
    fixtures.save_rows(root, rows)
    prepared = recovery.engine.prepare_source(root, 100)
    calls = []

    def client_factory(config, **kwargs):
        def transport(*args):
            calls.append(args)
            if len(calls) == 3:
                raise APIError("OpenAI HTTP 503; raw error retained in request log", response_body=json.dumps({
                    "error": {"type": "service_unavailable_error", "code": "server_is_overloaded"}}))
            return fixtures.envelope()
        kwargs["transport"] = transport
        return AstraClient(config, **kwargs)

    result = recovery.engine.execute_continuation(prepared, child, api_key="dummy",
        simulator_factory=lambda config: fixtures.FakeSimulator(config, keep_running=True),
        client_factory=client_factory, recorder_factory=fixtures.FakeRecorder)
    assert result["status"] == "infrastructure_error" and result["new_evaluated_actions"] == 10
    (child / "authorization.json").write_text(json.dumps({"approved_total_api_budget_usd": 100,
        "includes_prior_episode_usage": True, "authorization": "offline fixture approval"}))
    return root, child, result


def test_recovery_preserves_full_child_ledger_unknown_reservation_and_ordered_chain(tmp_path):
    root, child, terminal = make_chain(tmp_path)
    before = {str(p.relative_to(tmp_path)): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    prepared = recovery.prepare_recovery(child)
    assert prepared["offset"] == 15 and len(prepared["pairs"]) == 15
    assert prepared["original_budget"] == json.loads((child / "run_state.json").read_text())["budget"]
    assert prepared["handoff_budget"] == prepared["original_budget"]
    assert prepared["api_config"].max_cost_usd == 100
    assert prepared["original_budget"]["api_calls"] == 4
    assert prepared["original_budget"]["unknown_usage_calls"] == 1
    reason = prepared["recovery_reason"]
    assert reason["failed_request_ordinal"] == 4 and reason["next_request_ordinal"] == 5
    assert reason["retained_unknown_reservation_usd"] == 26.8644
    assert reason["next_reserved_total_usd"] == pytest.approx(terminal["budget_after"]["cost_usd_upper_bound"] + 26.8644)
    identity = recovery.recovery_identity(prepared)
    assert identity["parent_run"] == str(child)
    assert identity["root_run"] == str(root)
    assert [segment["action_count"] for segment in identity["chain"]] == [5, 10]
    assert identity["source_trace"]["segments"] == identity["chain"]
    assert identity["cost_cap_preserved"] == 100 and "cost_cap_change" not in identity
    assert {str(p.relative_to(tmp_path)): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before


@pytest.mark.parametrize("mutation", ["unknown_released", "cost_released", "ordinal", "parent_identity", "gripper", "approval", "error_type"])
def test_recovery_rejects_missing_usage_or_changed_chain_evidence(tmp_path, mutation):
    _, child, _ = make_chain(tmp_path)
    if mutation in ("unknown_released", "cost_released"):
        path = child / "run_state.json"
        data = json.loads(path.read_text())
        if mutation == "unknown_released":
            data["budget"]["unknown_usage_calls"] = 0
        else:
            data["budget"]["cost_usd_upper_bound"] -= 26.8644
        path.write_text(json.dumps(data))
    elif mutation == "ordinal":
        path = sorted((child / "api").glob("*.request.json"))[-1]
        data = json.loads(path.read_text())
        data["request_ordinal"] -= 1
        path.write_text(json.dumps(data))
    elif mutation == "parent_identity":
        path = child / "manifest.json"
        data = json.loads(path.read_text())
        data["configuration"]["parent_identity_hash"] = "changed"
        data["identity_hash"] = recovery.engine.recovery.canonical_hash(data["configuration"])
        path.write_text(json.dumps(data))
    elif mutation == "gripper":
        import gzip
        path = child / "continuation.jsonl.gz"
        with gzip.open(path, "rt") as stream:
            rows = [json.loads(line) for line in stream]
        rows[1]["action"][-1] = 1
        with gzip.open(path, "wt") as stream:
            stream.write("".join(json.dumps(row) + "\n" for row in rows))
    elif mutation == "approval":
        path = child / "authorization.json"
        data = json.loads(path.read_text())
        data["approved_total_api_budget_usd"] = 200
        path.write_text(json.dumps(data))
    else:
        path = sorted((child / "api").glob("*.result.json"))[-1]
        data = json.loads(path.read_text())
        data["error"] = "OpenAI HTTP 401"
        path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        recovery.prepare_recovery(child)


def test_existing_descendant_or_live_coordinator_blocks_stale_ledger_reuse(tmp_path, monkeypatch):
    root, child, _ = make_chain(tmp_path)
    monkeypatch.setattr(recovery, "active_coordinators", lambda paths: [{"pid": 123, "script": "recover_bowl_continuation.py"}])
    with pytest.raises(ValueError, match="competing"):
        recovery.prepare_recovery(child)
    monkeypatch.setattr(recovery, "active_coordinators", lambda paths: [])
    descendant = tmp_path / "spent-recovery"
    descendant.mkdir()
    (descendant / "manifest.json").write_text(json.dumps({"configuration": {"parent_run": str(child)}}))
    with pytest.raises(ValueError, match="descendant"):
        recovery.prepare_recovery(child)


def test_exact_combined_replay_gates_fresh_ordinal_and_retains_unknown_charge(tmp_path):
    _, child, old_result = make_chain(tmp_path)
    prepared = recovery.prepare_recovery(child)
    output = tmp_path / "recovered"
    simulators, calls = [], []

    class Simulator(fixtures.FakeSimulator):
        def step(self, action):
            self.keep_running = len(self.actions) + 1 <= 15
            return super().step(action)

    class Recorder(fixtures.FakeRecorder):
        def observe(self, obs, *, step):
            assert step == 15

    def sim_factory(config):
        simulator = Simulator(config, keep_running=True)
        simulators.append(simulator)
        return simulator

    def client_factory(config, **kwargs):
        def transport(*args):
            assert len(simulators[0].actions) == 15 and not simulators[0].closed
            proof = json.loads((output / "replay_validation.json").read_text())
            assert proof["status"] == "validated" and proof["completed_actions"] == 15
            assert proof["image_pairs_checked"] == 16
            calls.append(args)
            return fixtures.envelope()
        kwargs["transport"] = transport
        return AstraClient(config, **kwargs)

    result = recovery.execute_recovery(prepared, output, api_key="dummy", simulator_factory=sim_factory,
                                      client_factory=client_factory, recorder_factory=Recorder)
    assert result["status"] == "success" and result["new_evaluated_actions"] == 1
    assert result["cumulative_actions"] == 16 and result["reconstruction_actions"] == 15
    assert result["additional_api_calls"] == len(calls) == 1
    assert result["budget_after"]["api_calls"] == 5 and result["budget_after"]["unknown_usage_calls"] == 1
    assert result["budget_after"]["cost_usd_upper_bound"] > old_result["budget_after"]["cost_usd_upper_bound"]
    request = json.loads(next((output / "api").glob("*.request.json")).read_text())
    assert request["request_ordinal"] == 5 and request["configuration"]["max_cost_usd"] == 100
    identity = json.loads((output / "manifest.json").read_text())["configuration"]
    assert identity["kind"] == "same_episode_infrastructure_recovery"
    assert identity["parent_run"] == str(child) and len(identity["chain"]) == 2
    assert simulators[0].closed
    with pytest.raises(ValueError, match="descendant"):
        recovery.prepare_recovery(child)


def test_child_history_replay_mismatch_prevents_new_dispatch(tmp_path):
    _, child, _ = make_chain(tmp_path)
    prepared = recovery.prepare_recovery(child)

    class Simulator(fixtures.FakeSimulator):
        def step(self, action):
            obs, success = super().step(action)
            if len(self.actions) == 12:
                data = obs.to_dict()
                data["observation/state"][0] += 1e-8
                obs = PolicyObservation.from_dict(data)
            return obs, success

    def client_factory(config, **kwargs):
        kwargs["transport"] = lambda *args: pytest.fail("never dispatch after mismatching child history")
        return AstraClient(config, **kwargs)

    result = recovery.execute_recovery(prepared, tmp_path / "recovered", api_key="dummy",
        simulator_factory=lambda config: Simulator(config, keep_running=True), client_factory=client_factory)
    assert result["status"] == "infrastructure_error"
    assert result["new_evaluated_actions"] == result["additional_api_calls"] == 0
    assert result["budget_after"]["unknown_usage_calls"] == 1


def test_read_only_preflight_never_reads_credentials_or_starts_execution(tmp_path, monkeypatch):
    _, child, _ = make_chain(tmp_path)
    monkeypatch.setattr(recovery.getpass, "getpass", lambda *args: pytest.fail("no credential access in preflight"))
    monkeypatch.setattr(recovery, "execute_recovery", lambda *args, **kwargs: pytest.fail("no execution in preflight"))
    assert recovery.main(["--source", str(child), "--preflight"]) == 0
