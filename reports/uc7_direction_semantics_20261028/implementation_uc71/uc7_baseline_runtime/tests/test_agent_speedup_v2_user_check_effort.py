"""Explicit draft/executive effort and approval binding; no network or simulator."""
import pytest

from agent_speedup_v2_fakes import FakeV2Sim
from gpt_field.agent_speedup_v2_client import AgentSpeedUpV2Client, FixedAnswers, ScriptedV2Transport
from gpt_field.agent_speedup_v2_user_check import ReviewError
from gpt_field.agent_speedup_v2_user_check_agent_client import AgentFirstV2Client
from gpt_field.agent_speedup_v2_user_check_client import PlanLock, UserCheckV2Client
from gpt_field.astra import APIError
from test_agent_speedup_v2_user_check_runner import PLAN, TASK, runner, settings


def prepare_effort(tmp_path, effort="xhigh", *, response_effort=None):
    config, sim_config, monitor = settings()
    sim = FakeV2Sim(prompt=TASK.prompt)
    transport = ScriptedV2Transport(FixedAnswers(plans=[PLAN]), effort=response_effort or effort)
    store = runner.prepare_review(
        tmp_path / "review", config=config, sim_config=sim_config, task_key=TASK.key,
        episode_index=0, monitor_options=monitor, sim_factory=lambda _: sim,
        token_provider=lambda: "offline", transport=transport, video=False,
        reasoning_effort=effort)
    return store, sim, transport, config, sim_config, monitor


@pytest.mark.parametrize("effort", ["high", "xhigh"])
def test_actual_draft_and_all_executive_payloads_use_selected_effort(effort):
    draft = UserCheckV2Client(task_prompt=TASK.prompt, reasoning_effort=effort)
    lock = PlanLock(PLAN, task_prompt=TASK.prompt)
    executive = AgentFirstV2Client(
        task_prompt=TASK.prompt, approved_plan=PLAN, lock=lock, reasoning_effort=effort)
    request = {"fields": {}, "images": []}
    for client in (draft, executive):
        for mode in client._instruction_text:
            payload = client.build_payload(request, mode=mode)
            assert payload["reasoning"]["effort"] == effort
            assert payload["model"] == "gpt-6-astra"
    baseline = AgentSpeedUpV2Client(task_prompt=TASK.prompt)
    assert baseline.build_payload(request, mode="plan")["reasoning"]["effort"] == "medium"
    assert UserCheckV2Client(task_prompt=TASK.prompt).reasoning_effort == "high"
    assert AgentFirstV2Client(task_prompt=TASK.prompt, approved_plan=PLAN,
                              lock=lock).reasoning_effort == "high"


@pytest.mark.parametrize("effort", ["high", "xhigh"])
def test_draft_request_records_effort_without_policy_actions(tmp_path, effort):
    store, sim, transport, *_ = prepare_effort(tmp_path, effort)
    assert len(transport.calls) == 1
    assert transport.calls[0]["reasoning"]["effort"] == effort
    execution = store.load()["preparation"]["execution"]
    assert execution["reasoning_effort"] == effort
    assert execution["user_check"]["framework"]["model_reasoning_effort"] == effort
    assert sim.actions == []


def test_reported_effort_mismatch_is_rejected_without_fallback(tmp_path):
    with pytest.raises(APIError, match="reasoning effort differs"):
        prepare_effort(tmp_path, "xhigh", response_effort="high")
    assert not (tmp_path / "review/review.json").exists()
    assert len(list((tmp_path / "review/requests").glob("*.request.json"))) == 1


def test_effort_change_invalidates_review_before_execution(tmp_path):
    store, _, _, config, sim_config, monitor = prepare_effort(tmp_path)
    store.approve(expected_revision=0, reviewed_steps=[0])
    with pytest.raises(ReviewError, match="Runtime code changed"):
        runner.execute_reviewed(
            tmp_path / "run", review_dir=store.path, config=config, sim_config=sim_config,
            task_key=TASK.key, episode_index=0, monitor_options=monitor, video=False,
            token_provider=lambda: "offline", reasoning_effort="high")
    assert store.load()["status"] == "approved"
    assert not (tmp_path / "run").exists()


def test_execution_constructs_xhigh_client_and_records_actual_effort(tmp_path, monkeypatch):
    store, _, _, config, sim_config, monitor = prepare_effort(tmp_path)
    store.approve(expected_revision=0, reviewed_steps=[0])
    def execute(output, **kwargs):
        client = kwargs["client_factory"](config, task_prompt=TASK.prompt,
                                          token_provider=lambda: "offline")
        assert client.build_payload({"fields": {}, "images": []}, mode="agent")["reasoning"] == {
            "effort": "xhigh"}
        return {"reasoning_effort": client.reasoning_effort}
    def normalize(output, summary, review, lock, **kwargs):
        assert kwargs["client_metadata"]["reasoning_effort"] == "xhigh"
        assert review["preparation"]["execution"]["reasoning_effort"] == "xhigh"
        return summary
    monkeypatch.setattr(runner.base, "execute", execute)
    monkeypatch.setattr(runner, "_normalize_artifacts", normalize)
    result = runner.execute_reviewed(
        tmp_path / "run", review_dir=store.path, config=config, sim_config=sim_config,
        task_key=TASK.key, episode_index=0, monitor_options=monitor, video=False,
        token_provider=lambda: "offline", reasoning_effort="xhigh")
    assert result["reasoning_effort"] == "xhigh"


def test_run_cli_restores_reviewed_effort(tmp_path, monkeypatch):
    import gpt_field.chatgpt_auth as auth
    store, *_ = prepare_effort(tmp_path)
    store.approve(expected_revision=0, reviewed_steps=[0])
    monkeypatch.setattr(runner, "_live_gate", lambda *args: None)
    class NoCredentials:
        def __init__(self, **kwargs):
            pass
        def access_token(self):
            pytest.fail("No credentials are read in this offline wiring test")
    monkeypatch.setattr(auth, "SubscriptionCredentials", NoCredentials)
    calls = []
    def execute(output, **kwargs):
        calls.append(kwargs)
        return {"status": "fake", "output": str(output), "evaluation_completed": True,
                "user_check": {"review_id": store.load()["id"]}}
    monkeypatch.setattr(runner, "execute_reviewed", execute)
    assert runner.main(["run", "--review-dir", str(store.path), "--output", str(tmp_path / "run")]) == 0
    assert calls[0]["reasoning_effort"] == "xhigh"


def test_preflight_and_runtime_identity_include_selected_effort():
    config, sim_config, monitor = settings()
    result = runner.preflight(config, sim_config, task_key=TASK.key, episode_index=0,
                              monitor_options=monitor, reasoning_effort="xhigh")
    assert result["client"]["reasoning_effort"] == "xhigh"
    assert result["preparation_client"]["reasoning_effort"] == "xhigh"
    assert result["framework"]["model_reasoning_effort"] == "xhigh"
    assert "--reasoning-effort xhigh" in result["live_command"]
    assert result["live_command_first_run"] == result["live_command"]
    assert not result["simulator_started"] and result["api_calls"] == 0
    high = runner.runtime_inventory(config, sim_config, monitor_options=monitor)
    xhigh = result["runtime_sha256"]
    assert high["@configuration"] != xhigh["@configuration"]
    assert {k: v for k, v in high.items() if not k.startswith("@")} == {
        k: v for k, v in xhigh.items() if not k.startswith("@")}


def test_invalid_effort_fails_before_simulator_start(tmp_path):
    config, sim_config, monitor = settings()
    constructed = []
    with pytest.raises(ValueError, match="high or xhigh"):
        runner.prepare_review(
            tmp_path / "review", config=config, sim_config=sim_config, task_key=TASK.key,
            episode_index=0, monitor_options=monitor, reasoning_effort="extra-high",
            sim_factory=lambda _: constructed.append(True))
    assert not constructed and not (tmp_path / "review").exists()
