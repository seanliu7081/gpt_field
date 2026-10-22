"""Offline contract and lifecycle tests; no real model, credentials or physics."""
from contextlib import nullcontext
import copy
import json
from pathlib import Path

import pytest

from .client import CodexPolicyClient, ProtocolError
from .offline_check import fixture_request
from .policy import MODEL, PROVIDER


class FakeRPC:
    status = "completed"
    duplicate = False
    missing = False
    compact = False
    wrong_thread = False
    wrong_model = False
    instances = []

    def __init__(self, command, **kwargs):
        self.events = []
        self.sent = []
        self.calls = []
        self.closed = False
        self.turn = 0
        self.instances.append(self)

    def send(self, message):
        self.sent.append(message)

    def call(self, method, params):
        self.calls.append((method, copy.deepcopy(params)))
        if method == "initialize":
            return {}
        if method in ("thread/start", "thread/resume"):
            return {"model": "wrong" if self.wrong_model else MODEL, "modelProvider": PROVIDER,
                    "reasoningEffort": "medium", "thread": {"id": "persistent-1"}, "instructionSources": []}
        if method == "turn/start":
            self.turn += 1
            ident = f"turn-{self.turn}"
            event = {"id": 100 + self.turn, "method": "item/tool/call", "params": {
                "threadId": "foreign" if self.wrong_thread else "persistent-1", "turnId": ident,
                "namespace": "robot", "tool": "execute_eef_target", "callId": f"call-{self.turn}",
                "arguments": {"target": [0, 0, 1.01, 1, 0, 0, 0, 0], "steps": 1}}}
            if not self.missing:
                self.events.append(event)
            if self.duplicate:
                self.events.append(copy.deepcopy(event))
            if self.compact:
                self.events.append({"method": "thread/compacted", "params": {}})
            self.events.append({"method": "turn/completed", "params": {"turn": {"id": ident, "status": self.status}}})
            return {"turn": {"id": ident}}
        raise AssertionError(method)

    def next_event(self):
        return self.events.pop(0)

    def close(self):
        self.closed = True


def make_client(tmp_path, cls=FakeRPC, provider=lambda: "offline-oauth-fixture"):
    return CodexPolicyClient(token_provider=provider, rpc_factory=cls,
                             log_dir=tmp_path / "rpc", binary="fake-codex")


def feedback(command):
    from gpt_field.astranomy_agentic_motion import ShortTargetExecutor
    state = [0, 0, 1, 0, 0, 0, .04, -.04]
    executor = ShortTargetExecutor(state, **command["arguments"])
    action = executor.next_action(state)
    executor.observe_transition(state, action, state)
    return executor.feedback(state)


def test_one_thread_and_actual_feedback_on_next_turn(tmp_path):
    client = make_client(tmp_path)
    obs = fixture_request()
    command = client.predict(obs)
    rpc = client.rpc
    assert client.thread_id == "persistent-1"
    ack = next(x for x in rpc.sent if "result" in x)
    assert "No robot action has executed" in ack["result"]["contentItems"][0]["text"]
    with pytest.raises(ProtocolError):
        client.predict(obs)
    client.record_result(feedback(command))
    obs["step"] = 1
    client.predict(obs)
    assert len([m for m, _ in rpc.calls if m == "thread/start"]) == 1
    turns = [p for m, p in rpc.calls if m == "turn/start"]
    assert "actual_execution_feedback" in turns[1]["input"][0]["text"]
    assert len([i for i in turns[1]["input"] if i["type"] == "image"]) == 2
    assert turns[0]["threadId"] == turns[1]["threadId"]
    client.close()
    assert rpc.closed


@pytest.mark.parametrize("flag,value", [("duplicate", True), ("missing", True), ("compact", True),
    ("wrong_thread", True), ("wrong_model", True), ("status", "failed"), ("status", "interrupted")])
def test_invalid_turn_never_returns_executable_proposal(tmp_path, flag, value):
    cls = type("BadRPC", (FakeRPC,), {flag: value})
    client = make_client(tmp_path, cls)
    with pytest.raises(ProtocolError):
        client.predict(fixture_request())
    assert client.proposals == 0
    client.close()


def test_token_refresh_resumes_same_thread(tmp_path):
    values = iter(["offline-token-one", "offline-token-two"])
    client = make_client(tmp_path, provider=lambda: next(values))
    proposal = client.predict(fixture_request())
    old = client.rpc
    client.record_result(feedback(proposal))
    client.predict(fixture_request())
    assert old.closed and client.rpc is not old
    assert any(m == "thread/resume" and p["threadId"] == "persistent-1" for m, p in client.rpc.calls)
    assert client.token_rotations == 1
    client.close()


def test_no_api_key_fallback(tmp_path):
    client = make_client(tmp_path, provider=lambda: "sk-not-allowed")
    with pytest.raises(ValueError, match="API keys"):
        client.predict(fixture_request())
    assert client.rpc is None


@pytest.mark.parametrize("field", ["libero_success", "reward", "object_positions", "collision"])
def test_privileged_sensor_fields_rejected(tmp_path, field):
    client = make_client(tmp_path)
    request = fixture_request()
    request[field] = True
    with pytest.raises(ValueError):
        client.predict(request)
    assert client.rpc is None


def test_privileged_feedback_rejected(tmp_path):
    client = make_client(tmp_path)
    command = client.predict(fixture_request())
    result = feedback(command)
    result["libero_success"] = True
    with pytest.raises(ValueError):
        client.record_result(result)
    client.close()


@pytest.fixture
def runner(monkeypatch):
    from . import run
    monkeypatch.setattr(run, "freeze_sources", lambda path: None)
    monkeypatch.setattr(run, "source_manifest", lambda: {})
    monkeypatch.setattr(run.life, "DEFAULT_EXPECTED_SHA256", {})
    monkeypatch.setattr(run.life, "termination_as_interrupt", nullcontext)
    monkeypatch.setattr(run.life, "_simulator_without_credentials", nullcontext)
    return run


class FakeSim:
    def __init__(self, config, success_after=2):
        from gpt_field.observation import PolicyObservation
        request = fixture_request()
        self.obs = PolicyObservation(request["external_image"], request["wrist_image"],
            (0, 0, 1, 0, 0, 0, .04, -.04), "open the top drawer and put the bowl inside", (0.,) * 7, (0.,) * 7)
        self.status = {}
        self.metadata = {"fake": True}
        self.actions = []
        self.success_after = success_after
        self.closed = False

    def start(self):
        return self

    def reset(self, task_id, episode_index):
        assert (task_id, episode_index) == (3, 0)
        self.status = {"terminated": False, "libero_success": False}
        return self.obs

    def geometry(self):
        return fixture_request()["geometry"]

    def step(self, action):
        self.actions.append(action)
        success = len(self.actions) >= self.success_after
        self.status = {"terminated": success, "libero_success": success,
                       "collision": {"detected": True}}
        return self.obs, success

    def close(self):
        self.closed = True


def test_runner_checks_raw_success_and_cleans_up(tmp_path, runner):
    sims, clients = [], []
    def make_sim(config):
        sim = FakeSim(config)
        sims.append(sim)
        return sim
    def make_policy(config, **kwargs):
        client = CodexPolicyClient(config, rpc_factory=FakeRPC, binary="fake", **kwargs)
        clients.append(client)
        return client
    result = runner.execute(tmp_path / "run", task_key="drawer_bowl", token_provider=lambda: "offline-token",
                            video=False, sim_factory=make_sim, client_factory=make_policy)
    assert result["status"] == "success" and result["steps"] == 2
    assert result["libero_success"] and result["completion_time_eligible"]
    assert result["api_calls"] is None  # Do not mislabel Codex turns as requests.
    assert sims[0].closed and clients[0].rpc.closed
    assert result["collision"]["detected"]  # Record-only, not a disqualifier.


def test_failed_turn_executes_zero_robot_steps(tmp_path, runner):
    sim = FakeSim(None)
    class FailingRPC(FakeRPC):
        status = "failed"
    def make_policy(config, **kwargs):
        return CodexPolicyClient(config, rpc_factory=FailingRPC, binary="fake", **kwargs)
    result = runner.execute(tmp_path / "run", task_key="drawer_bowl", token_provider=lambda: "offline-token",
                            video=False, sim_factory=lambda _: sim, client_factory=make_policy)
    assert result["status"] == "infrastructure_error" and result["steps"] == 0
    assert not sim.actions and sim.closed and not result["completion_time_eligible"]


def test_preflight_and_missing_approval_cannot_launch(runner, monkeypatch, capsys):
    monkeypatch.setattr(runner, "execute", lambda *a, **k: pytest.fail("must not launch"))
    assert runner.main([]) == 0
    assert json.loads(capsys.readouterr().out)["full_test_started"] is False
    with pytest.raises(SystemExit) as err:
        runner.main(["--run"])
    assert err.value.code == 2


def test_no_episode_caps_or_wrong_task(runner):
    config = runner.sim_configuration("drawer_bowl")
    assert config.max_steps is None and config.stop_on_success and not config.stop_on_collision
    api = runner.configuration()
    assert api.max_api_calls is None and api.max_wall_time_s is None and api.max_cost_usd is None


def test_existing_run_never_overwritten(tmp_path, runner):
    with pytest.raises(FileExistsError):
        runner.execute(tmp_path, task_key="drawer_bowl", token_provider=lambda: "offline-token")
