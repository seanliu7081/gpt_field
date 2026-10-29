"""Offline overload recovery: same observation, complete response, one action stream."""

import json

import pytest

from gpt_field.astra import APIError, AstraConfig, BudgetExceeded, SchemaError
from gpt_field.chatgpt_plan import _protocol_error
from gpt_field.speed_first_hybrid import ChatGPTPlanSpeedFirstClient, SpeedFirstHybridPolicy
from test_distance_policy import intent, tick
from test_field_history import FakeTransport, envelope, observation


TOKEN = "offline-overload-oauth-secret"


class Clock:
    def __init__(self):
        self.now = 100.
        self.waits = []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.waits.append(seconds)
        self.now += seconds


@pytest.fixture
def clock(monkeypatch):
    import gpt_field.speed_first_hybrid as module
    value = Clock()
    monkeypatch.setattr(module.time, "sleep", value.sleep)
    monkeypatch.setattr(module.random, "uniform", lambda low, high: .25)
    return value


def overloaded(*, status=None, retry_after=None):
    error = _protocol_error("provider details " + TOKEN, http_status=status,
                            error_code="server_is_overloaded")
    if retry_after is not None:
        error.retry_after_s = retry_after
    return error


def response(raw=None):
    return envelope(intent(direct=True, phase="grasp") if raw is None else raw,
                    model="gpt-6-astra", reasoning={"effort": "medium"})


def client(transport, clock, **kwargs):
    return ChatGPTPlanSpeedFirstClient(token_provider=lambda: TOKEN, transport=transport,
                                       clock=clock, **kwargs)


@pytest.mark.parametrize("status", [None, 503])
def test_overload_retries_identical_payload_and_commits_once(status, clock, tmp_path):
    accepted = response()
    transport = FakeTransport(overloaded(status=status), accepted)
    instance = client(transport, clock, log_dir=tmp_path)
    result = instance.predict(observation().to_dict())
    assert result["controller"] == "contact_direct"
    assert transport.calls[0] == transport.calls[1]
    assert instance.history_snapshot() == transport.calls[1]["input"] + accepted["output"]
    stats = instance.history_stats()
    assert stats["observations"] == 1 and stats["feedback_records"] == 0
    assert stats["overload_retry_requests"] == 1
    assert stats["overload_retry_wait_s"] == pytest.approx(2.25)
    assert clock.waits == [2.25]
    budget = instance.budget_snapshot()
    assert budget["api_calls"] == 2 and budget["unknown_usage_calls"] == 1
    assert budget["input_tokens"] == 100 and budget["output_tokens"] == 50
    assert budget["elapsed_s"] == pytest.approx(2.25)
    requests = [json.loads(path.read_text()) for path in sorted(tmp_path.glob("*.request.json"))]
    assert [item["attempt"] for item in requests] == [1, 2]
    assert requests[0]["request_key"] == requests[1]["request_key"]
    assert all(item["overload_retry_policy"] for item in requests)
    retries = [json.loads(path.read_text()) for path in tmp_path.glob("*.overload_retry.json")]
    assert len(retries) == 1
    assert retries[0]["status"] == "bounded_retry_scheduled"
    assert retries[0]["attempt"] == 1 and retries[0]["next_attempt"] == 2
    assert retries[0]["wait_s"] == pytest.approx(2.25)
    assert retries[0]["same_observation"] is True
    assert TOKEN not in "".join(path.read_text() for path in tmp_path.iterdir())


def test_retries_preserve_existing_feedback_and_return_five_actions_once(clock, monkeypatch):
    import gpt_field.hybrid_policy as policy_module
    monkeypatch.setattr(policy_module.time, "monotonic", clock)
    first = intent(direct=True, phase="grasp")
    second = intent(direct=True, phase="recover")
    second["direct"]["actions"] = [[.01 * i, 0., -.02, 0., 0., 0., -1.] for i in range(5)]
    transport = FakeTransport(response(first), overloaded(), response(second))
    instance = client(transport, clock)
    policy = SpeedFirstHybridPolicy(instance)
    obs = observation()
    for _ in range(5):
        obs, _ = tick(policy, obs)
    preserved = instance.history_snapshot()
    actual = []
    for index in range(5):
        obs, decision = tick(policy, obs)
        actual.append(list(decision.action))
        if index == 0:
            assert decision.diagnostics["api_latency_s"] == pytest.approx(2.25)
    assert actual == second["direct"]["actions"]
    assert transport.calls[1] == transport.calls[2]
    assert instance.history_stats()["observations"] == 2
    assert instance.history_stats()["feedback_records"] == 10
    assert policy.step_index == 10 and len(transport.calls) == 3
    # The retry sends the same compacted history; the old raw history is neither
    # extended nor altered before that accepted response is committed.
    assert preserved[0] == instance.history_snapshot()[0]
    assert instance.budget_snapshot()["unknown_usage_calls"] == 1


def test_three_overloads_stop_without_history_or_executable_output(clock, tmp_path):
    transport = FakeTransport(overloaded(), overloaded(), overloaded(), response())
    instance = client(transport, clock, log_dir=tmp_path)
    with pytest.raises(APIError) as caught:
        instance.predict(observation().to_dict())
    assert caught.value.error_code == "server_is_overloaded"
    assert TOKEN not in str(caught.value)
    assert len(transport.calls) == 3 and len(transport.responses) == 1
    assert transport.calls[0] == transport.calls[1] == transport.calls[2]
    assert clock.waits == [2.25, 4.25]
    assert instance.last_response is None and instance.history_snapshot() == []
    assert instance.history_stats()["observations"] == 0
    assert instance.history_stats()["overload_retry_requests"] == 2
    assert instance.history_stats()["overload_retry_wait_s"] == pytest.approx(6.5)
    assert instance.budget_snapshot()["unknown_usage_calls"] == 3
    assert len(list(tmp_path.glob("*.request.json"))) == 3
    assert len(list(tmp_path.glob("*.result.json"))) == 3
    assert len(list(tmp_path.glob("*.overload_retry.json"))) == 2
    assert not list(tmp_path.glob("*.response.json"))


def test_field_retry_keeps_grounding_observation_and_executes_one_local_action(clock):
    transport = FakeTransport(overloaded(), response(intent(goal=(.06, 0., 0.))))
    instance = client(transport, clock)
    policy = SpeedFirstHybridPolicy(instance)
    _, decision = tick(policy, observation())
    assert decision.diagnostics["controller"] == "coarse_field"
    assert decision.diagnostics["plan"]["search"]["method"] == "dijkstra_distance_field"
    assert policy.step_index == 1
    assert instance.history_stats()["observations"] == 1
    assert instance.history_stats()["feedback_records"] == 1
    assert instance._prediction_observation is None
    assert instance.last_compact_prediction is not None
    assert transport.calls[0] == transport.calls[1]


def test_exhausted_retries_preserve_previous_observations_and_feedback(clock):
    transport = FakeTransport(response(), overloaded(), overloaded(), overloaded())
    instance = client(transport, clock)
    policy = SpeedFirstHybridPolicy(instance)
    obs = observation()
    for _ in range(5):
        obs, _ = tick(policy, obs)
    before = instance.history_snapshot()
    with pytest.raises(APIError):
        policy.act(obs)
    assert instance.history_snapshot() == before
    assert instance.history_stats()["observations"] == 1
    assert instance.history_stats()["feedback_records"] == 5
    assert policy.step_index == 5 and policy._pending is None
    assert instance.last_compact_prediction is None
    assert instance._prediction_observation is None


@pytest.mark.parametrize("code,status", [
    ("invalid_grant", 401), ("invalid_grant", 403),
    ("subscription_sharing_usage_limit_exceeded", 429),
    ("stream_missing_completion", None), ("stream_invalid_json", None),
    ("subscription_network_error", None), ("server_error", 503),
    ("server_is_overloaded", 401), ("server_is_overloaded", 429),
    ("server_is_overloaded", 500),
])
def test_only_the_explicit_overload_failure_is_retryable(code, status, clock):
    transport = FakeTransport(_protocol_error("failed", http_status=status, error_code=code), response())
    instance = client(transport, clock)
    with pytest.raises(APIError):
        instance.predict(observation().to_dict())
    assert len(transport.calls) == 1 and not clock.waits
    assert not instance.history_snapshot()


def test_schema_failure_after_completed_response_is_not_retried(clock):
    transport = FakeTransport(response({"invalid": "shape"}), response())
    instance = client(transport, clock)
    with pytest.raises(SchemaError):
        instance.predict(observation().to_dict())
    assert len(transport.calls) == 1 and not clock.waits
    assert instance.last_response is not None and not instance.history_snapshot()
    assert instance.budget_snapshot()["unknown_usage_calls"] == 0


def test_completed_response_cannot_trigger_retry_by_overload_code(clock, monkeypatch):
    transport = FakeTransport(response(), response())
    instance = client(transport, clock)

    def reject(*args):
        raise overloaded()

    monkeypatch.setattr(instance, "validate_response", reject)
    with pytest.raises(APIError):
        instance.predict(observation().to_dict())
    assert len(transport.calls) == 1 and not clock.waits
    assert instance.last_response is not None and not instance.history_snapshot()


def test_observation_commit_guard_prevents_retry(clock):
    def transport(*args):
        calls.append(1)
        instance._observation_count += 1
        raise overloaded()

    calls = []
    instance = client(transport, clock)
    with pytest.raises(APIError):
        instance.predict(observation().to_dict())
    assert len(calls) == 1 and not clock.waits


def test_call_budget_is_checked_before_retry_wait(clock):
    transport = FakeTransport(overloaded(), response())
    instance = client(transport, clock, config=AstraConfig(max_api_calls=1))
    with pytest.raises(BudgetExceeded):
        instance.predict(observation().to_dict())
    assert len(transport.calls) == 1 and not clock.waits
    assert instance.budget_snapshot()["api_calls"] == 1


def test_wall_budget_too_short_for_backoff_does_not_sleep_or_retry(clock):
    transport = FakeTransport(overloaded(), response())
    instance = client(transport, clock, config=AstraConfig(max_wall_time_s=1.))
    # The original provider error remains the terminal cause: the wall limit
    # has not elapsed, but there is insufficient room to schedule a retry.
    with pytest.raises(APIError) as caught:
        instance.predict(observation().to_dict())
    assert caught.value.error_code == "server_is_overloaded"
    assert len(transport.calls) == 1 and not clock.waits


def test_retry_after_is_respected_within_bounded_wait(clock):
    transport = FakeTransport(overloaded(status=503, retry_after=10.), response())
    instance = client(transport, clock)
    instance.predict(observation().to_dict())
    assert clock.waits == [10.]
    assert instance.history_stats()["overload_retry_wait_s"] == pytest.approx(10.)


def test_retry_after_over_sixty_seconds_stops_instead_of_retrying_early(clock):
    transport = FakeTransport(overloaded(status=503, retry_after=61.), response())
    instance = client(transport, clock)
    with pytest.raises(APIError):
        instance.predict(observation().to_dict())
    assert len(transport.calls) == 1 and not clock.waits


def test_reset_history_resets_retry_counters_not_lifetime_request_ledger(clock):
    instance = client(FakeTransport(overloaded(), response()), clock)
    instance.predict(observation().to_dict())
    instance.reset_history()
    stats = instance.history_stats()
    assert stats["overload_retry_requests"] == 0
    assert stats["overload_retry_wait_s"] == 0
    assert stats["observations"] == 0 and instance.history_snapshot() == []
    assert instance.budget_snapshot()["api_calls"] == 2


def test_keyboard_interrupt_during_backoff_never_sends_another_request(clock, monkeypatch):
    import gpt_field.speed_first_hybrid as module

    def interrupted(seconds):
        raise KeyboardInterrupt()

    monkeypatch.setattr(module.time, "sleep", interrupted)
    transport = FakeTransport(overloaded(), response())
    instance = client(transport, clock)
    with pytest.raises(KeyboardInterrupt):
        instance.predict(observation().to_dict())
    assert len(transport.calls) == 1 and instance.history_snapshot() == []
