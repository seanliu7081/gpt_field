import copy
import importlib.util
import json
from pathlib import Path

import pytest

from gpt_field.astra import AstraClient
from gpt_field.observation import PolicyObservation

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


budget = load("continue_bowl_chain_budget", ROOT / "scripts/continue_bowl_chain_budget.py")
history = load("recovery_fixtures_for_budget", ROOT / "tests/test_bowl_recovery.py")
fixtures = history.fixtures


def make_three_parts(tmp_path):
    root, child, _ = history.make_chain(tmp_path)
    current = tmp_path / "current"
    prepared = history.recovery.prepare_recovery(child)

    class Recorder(fixtures.FakeRecorder):
        def observe(self, obs, *, step):
            assert step == 15

    def client_factory(config, **kwargs):
        def transport(*args):
            response = fixtures.envelope()
            response["usage"].update(input_tokens=1_000_000, output_tokens=8192)
            return response
        kwargs["transport"] = transport
        return AstraClient(config, **kwargs)

    result = history.recovery.execute_recovery(prepared, current, api_key="dummy",
        simulator_factory=lambda config: fixtures.FakeSimulator(config, keep_running=True),
        client_factory=client_factory, recorder_factory=Recorder)
    assert result["status"] == "budget_exceeded" and result["new_evaluated_actions"] == 10
    authorization = tmp_path / "approved150.json"
    authorization.write_text(json.dumps({"authorization_text": "make budget to 150",
        "approved_total_api_budget_usd": 150, "supersedes_approved_total_api_budget_usd": 100,
        "includes_prior_episode_usage": True, "includes_unresolved_request_reservations": True,
        "max_actions": None, "root_run": root.name, "current_run": current.name}))
    return root, child, current, authorization, result


def test_three_part_budget_handoff_changes_only_cap_and_preserves_unknown_usage(tmp_path):
    root, child, current, auth, result = make_three_parts(tmp_path)
    before = {str(p.relative_to(tmp_path)): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    prepared = budget.prepare_budget_handoff(current, 150, auth)
    original = json.loads((current / "run_state.json").read_text())["budget"]
    expected = copy.deepcopy(original)
    expected["configuration"]["max_cost_usd"] = 150
    expected["max_cost_usd"] = 150
    assert prepared["original_budget"] == original and prepared["handoff_budget"] == expected
    assert prepared["offset"] == 25 and len(prepared["pairs"]) == 25
    assert prepared["original_budget"]["api_calls"] == 6
    assert prepared["original_budget"]["unknown_usage_calls"] == 1
    assert prepared["original_budget"]["cost_usd_upper_bound"] == result["budget_after"]["cost_usd_upper_bound"]
    identity = budget.budget_identity(prepared)
    assert identity["kind"] == "same_episode_chain_budget_continuation"
    assert identity["cost_cap_change"]["old_total_usd"] == 100
    assert identity["cost_cap_change"]["new_total_usd"] == 150
    assert identity["cost_cap_change"]["only_configuration_change"] == "max_cost_usd"
    assert [entry["action_count"] for entry in identity["chain"]] == [5, 10, 10]
    assert identity["parent_run"] == str(current) and identity["root_run"] == str(root)
    assert identity["authorization_source"]["path"] == str(auth)
    assert {str(p.relative_to(tmp_path)): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before


@pytest.mark.parametrize("cap", [None, 100, 149, 151, float("inf"), float("nan"), True])
def test_exact_new_total_must_be_explicitly_150(cap):
    with pytest.raises(ValueError, match="explicit"):
        budget.prepare_budget_handoff("unused", cap, "unused")


@pytest.mark.parametrize("mutation", ["live", "approval", "unknown", "cost", "identity", "gripper", "descendant"])
def test_invalid_terminal_chain_or_authorization_fails_closed(tmp_path, mutation):
    _, _, current, auth, _ = make_three_parts(tmp_path)
    if mutation in ("live", "unknown", "cost"):
        path = current / "run_state.json"
        data = json.loads(path.read_text())
        if mutation == "live":
            data["status"] = "running"
        elif mutation == "unknown":
            data["budget"]["unknown_usage_calls"] = 0
        else:
            data["budget"]["cost_usd_upper_bound"] -= 26.8644
        path.write_text(json.dumps(data))
    elif mutation == "approval":
        data = json.loads(auth.read_text())
        data["includes_unresolved_request_reservations"] = False
        auth.write_text(json.dumps(data))
    elif mutation == "identity":
        path = current / "manifest.json"
        data = json.loads(path.read_text())
        data["configuration"]["parent_identity_hash"] = "changed"
        data["identity_hash"] = budget.engine.recovery.canonical_hash(data["configuration"])
        path.write_text(json.dumps(data))
    elif mutation == "gripper":
        import gzip
        path = current / "continuation.jsonl.gz"
        with gzip.open(path, "rt") as stream:
            rows = [json.loads(line) for line in stream]
        rows[1]["action"][-1] = 1
        with gzip.open(path, "wt") as stream:
            stream.write("".join(json.dumps(row) + "\n" for row in rows))
    else:
        descendant = tmp_path / "already-spent"
        descendant.mkdir()
        (descendant / "manifest.json").write_text(json.dumps({"configuration": {"parent_run": str(current)}}))
    with pytest.raises(ValueError):
        budget.prepare_budget_handoff(current, 150, auth)


def test_full_three_part_replay_precedes_fresh_request_under150(tmp_path):
    _, _, current, auth, source_result = make_three_parts(tmp_path)
    prepared = budget.prepare_budget_handoff(current, 150, auth)
    output = tmp_path / "next"
    simulations, calls = [], []

    class Simulator(fixtures.FakeSimulator):
        def step(self, action):
            self.keep_running = len(self.actions) + 1 <= 25
            return super().step(action)

    class Recorder(fixtures.FakeRecorder):
        def observe(self, obs, *, step):
            assert step == 25

    def sim_factory(config):
        simulator = Simulator(config, keep_running=True)
        simulations.append(simulator)
        return simulator

    def client_factory(config, **kwargs):
        assert config.max_cost_usd == 150
        def transport(*args):
            assert len(simulations[0].actions) == 25 and not simulations[0].closed
            proof = json.loads((output / "replay_validation.json").read_text())
            assert proof["status"] == "validated" and proof["completed_actions"] == 25
            assert proof["image_pairs_checked"] == 26
            calls.append(args)
            return fixtures.envelope()
        kwargs["transport"] = transport
        return AstraClient(config, **kwargs)

    result = budget.execute_budget_handoff(prepared, output, api_key="dummy", simulator_factory=sim_factory,
                                          client_factory=client_factory, recorder_factory=Recorder)
    assert result["status"] == "success" and result["cumulative_actions"] == 26
    assert result["new_evaluated_actions"] == result["additional_api_calls"] == len(calls) == 1
    assert result["budget_after"]["api_calls"] == 7 and result["budget_after"]["unknown_usage_calls"] == 1
    assert result["budget_after"]["cost_usd_upper_bound"] > source_result["budget_after"]["cost_usd_upper_bound"]
    request = json.loads(next((output / "api").glob("*.request.json")).read_text())
    assert request["request_ordinal"] == 7 and request["configuration"]["max_cost_usd"] == 150
    assert simulations[0].closed
    with pytest.raises(ValueError, match="descendant"):
        budget.prepare_budget_handoff(current, 150, auth)


def test_current_segment_replay_drift_blocks_all_paid_dispatch(tmp_path):
    _, _, current, auth, _ = make_three_parts(tmp_path)
    prepared = budget.prepare_budget_handoff(current, 150, auth)

    class Simulator(fixtures.FakeSimulator):
        def step(self, action):
            obs, success = super().step(action)
            if len(self.actions) == 20:
                data = obs.to_dict()
                data["observation/state"][0] += 1e-8
                obs = PolicyObservation.from_dict(data)
            return obs, success

    def client_factory(config, **kwargs):
        kwargs["transport"] = lambda *args: pytest.fail("no inference after replay drift")
        return AstraClient(config, **kwargs)

    result = budget.execute_budget_handoff(prepared, tmp_path / "next", api_key="dummy",
        simulator_factory=lambda config: Simulator(config, keep_running=True), client_factory=client_factory)
    assert result["status"] == "infrastructure_error"
    assert result["additional_api_calls"] == result["new_evaluated_actions"] == 0
    assert result["budget_after"]["unknown_usage_calls"] == 1


def test_preflight_never_accesses_credentials_or_executes(tmp_path, monkeypatch):
    _, _, current, auth, _ = make_three_parts(tmp_path)
    monkeypatch.setattr(budget.getpass, "getpass", lambda *args: pytest.fail("no credential access"))
    monkeypatch.setattr(budget, "execute_budget_handoff", lambda *args, **kwargs: pytest.fail("no execution"))
    assert budget.main(["--source", str(current), "--authorization", str(auth),
                        "--max-total-cost-usd", "150", "--preflight"]) == 0
