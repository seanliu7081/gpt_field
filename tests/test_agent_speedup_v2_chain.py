"""RUN-EXT (builder BI; contract §C14.3, user decision 2026-10-19 live-first): the v2 runner's live mode, the
relation smoke, the stuck alert, the v2 trace and summary fields, the preflight payloads and the v2 video recorder.

No network, no simulator, no model: clients are the real v2 client over ``ScriptedV2Transport`` (offline mock) and
the worlds are the fakes' ``FakeV2WorldSim(FakeTwoViewSim(task2_like_scene()))``. The relation layer (BR, built
concurrently) is the policy tests' ``StandIn`` where the test is about the runner; the end-to-end chain with the
real relation layer skips until ``agent_speedup_v2_relations`` lands.
"""

from __future__ import annotations

import ast
import importlib.util
import inspect
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

import agent_speedup_v2_fakes as fakes
from agent_speedup_v2_fakes import FakeV2Sim, FakeV2WorldSim, truth_result
from test_agent_speedup_v2 import TASK2_PLAN, StandIn, answers
from gpt_field import agent_speedup_v2_thresholds as th
from gpt_field.agent_speedup_v2 import AgentSpeedUpV2Policy
from gpt_field.agent_speedup_v2_client import AgentSpeedUpV2Client, FixedAnswers, ScriptedV2Transport
from gpt_field.agent_speedup_v2_tasks import resolve_v2_task

ROOT = Path(__file__).resolve().parents[1]
RUNNER_FILE = ROOT / "scripts/evaluate_agent_speedup_v2.py"
spec = importlib.util.spec_from_file_location("evaluate_agent_speedup_v2_chain", RUNNER_FILE)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
CONFIG = runner.configuration()
T2 = resolve_v2_task("libero_10_t2")
LIBERO = "/workspace/heading_flow_pub/third_party/LIBERO"
VIDEO_FILE = ROOT / "src/gpt_field/agent_speedup_v2_video.py"


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


@pytest.fixture(scope="module")
def scene():
    return fakes.task2_like_scene()


def world_sim(scene, *, images=False, **kwargs):
    return FakeV2WorldSim(fakes.FakeTwoViewSim(scene, images=images), **kwargs)


def scripted_client_factory(scripted=None, seen=None):
    """The v2 client with an offline scripted transport, in the runner's construction signature."""
    def factory(config=None, *, token_provider, task_prompt, transport=None, log_dir=None):
        if seen is not None:
            seen.append({"token_provider": token_provider, "task_prompt": task_prompt, "transport": transport,
                         "log_dir": log_dir})
        return AgentSpeedUpV2Client(config, task_prompt=task_prompt, token_provider=token_provider,
                                    transport=ScriptedV2Transport(scripted or answers()), log_dir=log_dir)
    return factory


def standin_policy_factory(layer=None, seen=None):
    def factory(client, **kwargs):
        if seen is not None:
            seen.append(kwargs)
        return AgentSpeedUpV2Policy(client, relations=layer or StandIn())
    return factory


def live(tmp_path, sim, *, name="live", stop_after_steps=60, scripted=None, layer=None, **kwargs):
    output = tmp_path / name
    kwargs.setdefault("client_factory", scripted_client_factory(scripted))
    kwargs.setdefault("policy_factory", standin_policy_factory(layer))
    summary = runner.execute(output, config=CONFIG, sim_config=runner.build_sim_config("libero_10_t2")[0],
                             task_key="libero_10_t2", episode_index=0, token_provider=lambda: "offline-token",
                             sim_factory=lambda cfg: sim, video=False, stop_after_steps=stop_after_steps, **kwargs)
    return summary, output


# --------------------------------------------------------------------------------------- 1. live mode end to end
def test_live_run_writes_every_artifact_with_v2_trace_and_accounting(tmp_path, scene):
    sim = world_sim(scene)
    summary, output = live(tmp_path, sim)
    assert summary["status"] == "user_stop" and summary["steps"] == 60, summary["error"]
    assert summary["error"]["source"] == "step_cap" and summary["evaluation_scored"] is True
    for name in ("summary.json", "runner_summary.json", "task_evaluation.json", "episodes.jsonl", "run_state.json",
                 "ledger.json", "agent_stats.json", "manifest.json", "trace.jsonl", "client.json", "policy.json",
                 "physical_truth.json", "answer_key.json", "parse_check.json", "initial_observation_check.json"):
        assert (output / name).is_file(), name
    manifest = json.loads((output / "manifest.json").read_text())["configuration"]
    assert manifest["mode"] == runner.LIVE_MODE and manifest["kind"] == "agent_speedup_v2_full_task"
    assert manifest["billing"] == "chatgpt_subscription" and manifest["oracle"]["enabled"] is False
    assert "oracle" not in manifest["simulator_monitor"]
    assert manifest["operator_stop_after_steps"] == 60 and manifest["live_gate"] == runner.LIVE_GATE
    meta = manifest["agent_speedup_v2"]
    from gpt_field import agent_speedup_v2_client as client_module, agent_speedup_v2_plan as plan_module
    assert meta["instructions_sha256"] == client_module.instructions_sha256()
    assert meta["plan_schema_sha256"] == plan_module.schemas_sha256()
    assert meta["constants_sha256"] == th.constants_sha256()
    assert manifest["lint_check"]["require_all"] is True and meta["prompt_version"] == client_module.PROMPT_VERSION
    trace = rows(output / "trace.jsonl")
    names = [row["event"] for row in trace]
    assert names[:2] == ["reset", "threshold_set"]
    for event in ("relation_plan", "plan_validation", "step_start", "step_end", "relation_outcome", "action",
                  "executed", "feedback", "operator_stop"):
        assert event in names, event
    plan_row = next(row for row in trace if row["event"] == "relation_plan")
    assert plan_row["plan"] == TASK2_PLAN and plan_row["mode"] == runner.LIVE_MODE and "policy_step" in plan_row
    stop = next(row for row in trace if row["event"] == "operator_stop")
    assert stop["source"] == "step_cap" and stop["step"] == 60
    outcome_rows = [row for row in trace if row["event"] == "relation_outcome"]
    assert {row["outcome"]["relation"] for row in outcome_rows} >= {"calibrated", "registered", "turned",
                                                                   "supported_on"}
    calls = summary["model_calls"]
    assert calls["calls_by_kind"]["plan"] == 1 and calls["calls_by_kind"]["point_fine"] >= 1
    assert calls["runner_calls_by_kind"]["plan"] == 1 and set(calls["latency_s_by_kind"]) >= {"plan", "point_fine"}
    assert set(calls["tokens"]) >= {"input", "output", "reasoning", "output_by_kind", "visible_by_kind"}
    timing = summary["timing_components"]
    assert set(timing["model_pipeline_s_by_kind"]) == set(calls["latency_s_by_kind"])
    assert abs(sum(timing["model_pipeline_s_by_kind"].values()) - timing["model_pipeline_s"]) < 1e-3
    assert timing["policy_compute_s"] == pytest.approx(timing["decision_s"] - timing["model_pipeline_s"])
    assert summary["api_calls"] == sum(calls["calls_by_kind"].values())
    assert summary["relation_plan"] == TASK2_PLAN
    assert summary["v2_event_counts"]["relation_plan"] == 1 and "relation_plan" not in summary["nominal_events_missing"]
    assert summary["plan_steps"] and summary["step_budget"]["steps"] == 60
    assert summary["physical_truth"]["verdict"] == "raw_failure" and summary["v2_success"] is False
    check = json.loads((output / "parse_check.json").read_text())
    assert check["plan_available"] is True and check["plan_match"]["all_atoms_planned"] is True
    assert [row["planned"] for row in check["plan_match"]["atoms"]] == [True, True]
    assert [row["planned"] for row in check["atoms"]] == [True, True]          # copied from plan_match (review F10)
    assert check["planned_source"] == f"plan_match ({runner.PLAN_MATCH_VERSION})"
    build = manifest["live_build_check"]                                     # review F6: recorded, zero calls
    assert build["passed"] is True and set(build["payload_sha256"]) == {"plan", "point_fine", "point_coarse", "repair"}
    assert build["plan_schema_sha256"] == plan_module.schemas_sha256()
    assert build["instructions_sha256"] == client_module.instructions_sha256()
    policy = json.loads((output / "policy.json").read_text())
    assert policy["class"] == "AgentSpeedUpV2Policy" and policy["overrides"] == {}
    assert policy["relations_version"] == StandIn.RELATIONS_VERSION
    client = json.loads((output / "client.json").read_text())
    assert client["class"] == "AgentSpeedUpV2Client" and client["task_prompt"] == T2.prompt
    assert client["reasoning_effort"] == "medium" and client["transport"] == "ScriptedV2Transport"
    assert client["prompt_version"] == client_module.PROMPT_VERSION and client["offline_mock"] is True
    json.dumps(summary, allow_nan=False)


def test_live_success_scores_raw_libero_and_v2_success_needs_strict_truth(tmp_path, scene):
    held = truth_result(False, True, ["not_held", "at_rest"])
    settled = truth_result(True, True, [], state="after_release_settle")
    sim = FakeV2Sim(prompt=T2.prompt, success_at=30, terminal_truth=held, settled_truth=settled,
                    wrist=fakes.FakeTwoViewSim(scene, images=False))
    summary, output = live(tmp_path, sim, stop_after_steps=None)
    assert summary["status"] == "success" and summary["steps"] == 30
    assert summary["task_success"] is True and summary["evaluation_completed"] is True
    assert summary["libero_success"] is True and summary["physical_truth"]["verdict"] == "physical_success"
    assert summary["v2_success"] is True
    perched = truth_result(False, True, ["lowest_point_below_rim", "not_perched_on_rim"])
    sim = FakeV2Sim(prompt=T2.prompt, success_at=30, terminal_truth=perched, settled_truth=perched,
                    wrist=fakes.FakeTwoViewSim(scene, images=False))
    summary, _ = live(tmp_path, sim, name="lenient", stop_after_steps=None)
    assert summary["status"] == "success" and summary["task_success"] is True
    assert summary["physical_truth"]["verdict"] == "lenient_success" and summary["v2_success"] is False


def test_live_operator_stop_file_and_stop_after_steps_validation(tmp_path, scene):
    class StopAt(AgentSpeedUpV2Policy):
        def observe_transition(self, before, decision, after):
            feedback = super().observe_transition(before, decision, after)
            if self.step_index == 7:
                (tmp_path / "live" / runner.OPERATOR_STOP_FILE).write_text("{}")
            return feedback
    summary, output = live(tmp_path, world_sim(scene), stop_after_steps=None,
                           policy_factory=lambda client: StopAt(client, relations=StandIn()))
    assert summary["status"] == "user_stop" and summary["steps"] == 7 and summary["error"]["source"] == "file"
    with pytest.raises(ValueError, match="positive integer"):
        live(tmp_path, world_sim(scene), name="zero", stop_after_steps=0)
    with pytest.raises(ValueError, match="operator stop of a live run"):
        runner.execute(tmp_path / "smoke", config=CONFIG, sim_config=runner.build_sim_config("libero_10_t2")[0],
                       task_key="libero_10_t2", episode_index=0, sim_factory=lambda cfg: world_sim(scene),
                       video=False, smoke=True, smoke_steps=5, stop_after_steps=3)


def test_live_needs_credentials_and_the_client_gets_only_the_reset_sentence(tmp_path, scene):
    with pytest.raises(runner.MissingCredentials):
        runner.execute(tmp_path / "nocred", config=CONFIG, sim_config=runner.build_sim_config("libero_10_t2")[0],
                       task_key="libero_10_t2", episode_index=0, sim_factory=lambda cfg: world_sim(scene),
                       video=False, client_factory=scripted_client_factory(),
                       policy_factory=standin_policy_factory())
    seen, kwargs = [], []
    summary, _ = live(tmp_path, world_sim(scene), stop_after_steps=3,
                      client_factory=scripted_client_factory(seen=seen),
                      policy_factory=standin_policy_factory(seen=kwargs))
    assert summary["status"] == "user_stop"
    assert len(seen) == 1 and seen[0]["task_prompt"] == T2.prompt and seen[0]["transport"] is None
    assert kwargs == [{}]                                     # policy_factory(client): no task key, no overrides


def test_reset_sentence_mismatch_stops_before_any_model_call(tmp_path, scene):
    other = resolve_v2_task("libero_goal_g7").prompt
    sim = FakeV2WorldSim(fakes.FakeTwoViewSim(scene, images=False, prompt=other), prompt=T2.prompt)
    seen = []
    transport_answers = answers()
    summary, _ = live(tmp_path, sim, client_factory=scripted_client_factory(transport_answers, seen=seen))
    assert summary["status"] == "infrastructure_error" and sim.actions == [] and seen == []
    assert transport_answers.served == []


def test_live_api_error_is_a_subscription_error_and_truth_still_runs(tmp_path, scene):
    class Failing(FixedAnswers):
        def answer(self, kind, meta):
            raise runner.APIError("quota exhausted")
    summary, output = live(tmp_path, world_sim(scene), scripted=Failing(plans=[TASK2_PLAN]))
    assert summary["status"] == "subscription_error" and summary["steps"] >= 1
    assert summary["physical_truth"]["verdict"] != "truth_unavailable"
    assert (output / "physical_truth.json").is_file()


# ----------------------------------------------------------------------------------- 2. the ORACLE stays out of live
def test_oracle_is_refused_in_live_mode(tmp_path, scene):
    base = dict(config=CONFIG, sim_config=runner.build_sim_config("libero_10_t2")[0], task_key="libero_10_t2",
                episode_index=0, token_provider=lambda: "t", video=False,
                client_factory=scripted_client_factory(), policy_factory=standin_policy_factory())
    with pytest.raises(ValueError, match="ORACLE"):
        runner.execute(tmp_path / "a", sim_factory=lambda cfg: world_sim(scene),
                       monitor_options={"goal_aware": True, "truth_window": True, "oracle": True}, **base)
    with pytest.raises(ValueError, match="ORACLE"):
        runner.execute(tmp_path / "b", sim_factory=lambda cfg: world_sim(scene),
                       oracle_factory=lambda sim, key: None, **base)
    assert not (tmp_path / "a").exists() and not (tmp_path / "b").exists()
    sim = world_sim(scene)
    sim.metadata = {**sim.metadata, "oracle_enabled": True}
    summary = runner.execute(tmp_path / "c", sim_factory=lambda cfg: sim, stop_after_steps=2, **base)
    assert summary["status"] == "infrastructure_error" and summary["error"]["type"] == "ValueError"
    assert "ORACLE" in (tmp_path / "c" / runner.TRACEBACK_FILE).read_text() and sim.actions == []


def test_runner_never_loads_the_oracle_outside_the_relation_smoke():
    tree = ast.parse(RUNNER_FILE.read_text())
    module_level = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
    assert not any("oracle" in ast.unparse(node).lower() for node in module_level)
    code = ("import importlib.util, sys\n"
            f"spec = importlib.util.spec_from_file_location('r', {str(RUNNER_FILE)!r})\n"
            "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)\n"
            "cls = m._default_client_factory(False, live=True)\n"
            "policy = m._v2_policy_factory\n"
            "print(cls.__name__, sorted(k for k in sys.modules if 'oracle' in k.lower() or 'smoke_scenarios' in k))\n")
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=120,
                            env={"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin", "HOME": "/tmp"})
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.strip().splitlines()[-1] == "AgentSpeedUpV2Client []"


def test_default_v2_policy_factory_passes_no_overrides(monkeypatch):
    import gpt_field.agent_speedup_v2 as policy_module
    seen = []

    class Recording(AgentSpeedUpV2Policy):
        def __init__(self, client, **kwargs):
            seen.append(kwargs)
            super().__init__(client, relations=StandIn())
    monkeypatch.setattr(policy_module, "AgentSpeedUpV2Policy", Recording)
    client = AgentSpeedUpV2Client(CONFIG, task_prompt=T2.prompt, transport=ScriptedV2Transport(answers()))
    policy = runner._v2_policy_factory(client)
    assert isinstance(policy, Recording) and seen == [{}]
    source = inspect.getsource(runner._v2_policy_factory)
    assert "AgentSpeedUpV2Policy(client)" in source and "overrides" not in source.split('"""')[-1]


def test_default_factories_by_mode():
    assert runner._default_client_factory(True) is runner.OfflinePlumbingClient
    assert runner._default_client_factory(False, live=True) is AgentSpeedUpV2Client
    with pytest.raises(ValueError, match="explicit live run"):
        runner._default_client_factory(False)
    from gpt_field.agent_speedup_v2_video import AgentSpeedUpV2VideoRecorder
    assert runner._default_recorder_factory() is AgentSpeedUpV2VideoRecorder


@pytest.mark.skipif(importlib.util.find_spec("gpt_field.agent_speedup_v2_relations") is None,
                    reason="the relation layer (BR, Wave 2) has not landed")
def test_default_live_policy_is_the_real_v2_policy(tmp_path, scene):
    summary, output = live(tmp_path, world_sim(scene, images=True), stop_after_steps=2,
                           policy_factory=None)
    policy = json.loads((output / "policy.json").read_text())
    assert policy["class"] == "AgentSpeedUpV2Policy" and policy["overrides"] == {} and policy["episode_factory"] is None
    assert policy["relations_version"] is not None and summary["status"] == "user_stop", summary["error"]


# ------------------------------------------------------------------------------------------- 3. the relation smoke
class ReportingOracle(fakes.FakeOracleAnswers):
    def report(self):
        return {"counts": len(self.log), "answers": list(self.log)}


def relation_smoke(tmp_path, sim, *, name="relation", smoke_steps=40, policy_factory=None, oracle=None):
    built = {}

    def oracle_factory(simulator, key):
        built["answers"] = oracle or ReportingOracle(simulator.world, plan=TASK2_PLAN)
        built["key"] = key
        return built["answers"]
    output = tmp_path / name
    summary = runner.execute(output, config=CONFIG, sim_config=runner.build_sim_config("libero_10_t2")[0],
                             task_key="libero_10_t2", episode_index=0, sim_factory=lambda cfg: sim, video=False,
                             smoke=True, smoke_script="relation", smoke_steps=smoke_steps,
                             oracle_factory=oracle_factory,
                             policy_factory=policy_factory or standin_policy_factory())
    return summary, output, built


def test_relation_smoke_runs_the_v2_client_behind_the_oracle(tmp_path, scene):
    summary, output, built = relation_smoke(tmp_path, world_sim(scene))
    assert summary["status"] == "diagnostic_complete" and summary["steps"] == 40, summary["error"]
    assert built["key"] == "libero_10_t2" and built["answers"].log
    assert summary["api_calls"] == 0 and summary["mock_response_calls"] >= 2
    assert summary["model_calls"]["offline_mock"] is True and summary["model_calls"]["real_model_calls"] == 0
    assert summary["task_success"] is False and summary["evaluation_scored"] is False
    manifest = json.loads((output / "manifest.json").read_text())["configuration"]
    assert manifest["simulator_monitor"]["oracle"] is True and manifest["oracle"]["enabled"] is True
    assert manifest["smoke_script"] == "relation" and manifest["billing"] == "offline_mock"
    assert any(item["path"] == runner.ORACLE_PATH for item in manifest["runner_helpers"])
    assert (output / "frozen_source" / "agent_speedup_v2_oracle.py").is_file()
    client = json.loads((output / "client.json").read_text())
    assert client["class"] == "AgentSpeedUpV2Client" and client["offline_mock"] is True
    assert json.loads((output / "oracle_report.json").read_text())["counts"] == len(built["answers"].log)
    assert summary["oracle"]["report"] == "oracle_report.json"
    events = [row["event"] for row in rows(output / "trace.jsonl")]
    assert "relation_plan" in events and "step_start" in events


def test_smoke_v2_success_uses_the_live_rule(tmp_path, scene):
    """Lead decision L5: in a smoke, v2_success = raw LIBERO success AND strict physical truth, as live."""
    held = truth_result(False, True, ["not_held", "at_rest"])
    settled = truth_result(True, True, [], state="after_release_settle")
    sim = FakeV2Sim(prompt=T2.prompt, success_at=12, terminal_truth=held, settled_truth=settled,
                    wrist=fakes.FakeTwoViewSim(scene, images=False))
    sim.world = sim.wrist
    summary, _, _ = relation_smoke(tmp_path, sim)
    assert summary["status"] == "diagnostic_sim_terminated" and summary["libero_success"] is True
    assert summary["task_success"] is False and summary["v2_success"] is True
    assert summary["physical_truth"]["v2_success"] is True


@pytest.mark.skipif(importlib.util.find_spec("gpt_field.agent_speedup_v2_relations") is None,
                    reason="the relation layer (BR, Wave 2) has not landed")
def test_relation_smoke_end_to_end_with_the_real_relation_layer(tmp_path, scene):
    """§C14.3 item 9: FakeV2WorldSim + a fake ORACLE + the real policy, client and relations: LIBERO success from
    the fake world, strict physical truth, v2_success (L5) and every artifact."""
    sim = world_sim(scene, images=True)
    summary, output, _ = relation_smoke(tmp_path, sim, smoke_steps=1300, policy_factory=runner._v2_policy_factory,
                                        oracle=ReportingOracle(sim.world))
    assert summary["status"] == "diagnostic_sim_terminated", (summary["error"], summary["relation_outcome_counts"])
    assert summary["libero_success"] is True and summary["physical_truth"]["v2_success"] is True
    assert summary["v2_success"] is True and summary["steps"] <= 1300
    assert summary["api_calls"] == 0


# --------------------------------------------------------------------------------------------- 4. stuck alerts
def rec(n, relation, status, reason="r", pids=None):
    return {"n": n, "skill": relation, "relation": relation, "status": status, "reason": reason,
            "pids": pids if pids is not None else {"obj": "C"}}


def test_stuck_reason_criteria():
    assert runner._stuck_reason([]) is None
    window = [rec(i, "held", "failed", pids={"obj": f"P{i}"}) for i in range(10)] + [rec(10, "turned", "succeeded")]
    assert "10 of the last 11" in runner._stuck_reason(window)
    assert runner._stuck_reason(window[:9]) is None
    held = [rec(1, "held", "succeeded")] + [rec(i, "held", "failed") for i in range(2, 7)]
    assert runner._stuck_reason(held) is None                                   # 5 failures: not more than 5
    assert "re-establish 6 times" in runner._stuck_reason(held + [rec(7, "held", "failed")])
    other = [rec(1, "held", "succeeded")] + [rec(i, "held", "failed", pids={"obj": "A"}) for i in range(2, 9)]
    assert runner._stuck_reason(other) is None or "re-establish" not in runner._stuck_reason(other)
    never = [rec(i, "held", "failed") for i in range(1, 8)]
    assert runner._stuck_reason(never) is None                                  # never met: not a re-establishment
    assert runner._immediate_stuck_reason(rec(1, "plan", "rejected", "invalid_plan")) is not None
    assert runner._immediate_stuck_reason(rec(1, "plan", "failed", "invalid_plan")) is None
    alert = {"step": 670, "reason": "10 of the last 12 relation outcomes failed or were rejected"}
    assert runner._repeat_stuck_reason(runner.STUCK_REPEAT_FAILURES - 1, alert) is None
    assert runner._repeat_stuck_reason(runner.STUCK_REPEAT_FAILURES, None) is None
    assert runner._repeat_stuck_reason(runner.STUCK_REPEAT_FAILURES, alert).startswith(
        f"still stuck: {runner.STUCK_REPEAT_FAILURES} further") and "step 670" in runner._repeat_stuck_reason(12, alert)


def test_a_repoint_loop_alerts_although_every_lap_registers_the_part():
    """Integration (live smoke 003): each repair repoint records a succeeded "registered" outcome; the loop held
    failed / supported_on failed / registered succeeded never reached 10 of 12 failures. Registrations and the
    calibration are not counted."""
    lap = [rec(0, "held", "failed", pids={"obj": "C", "grip": ["D"]}),
           rec(0, "supported_on", "failed", pids={"obj": "C", "grip": ["D"], "ref": "B"}),
           rec(0, "registered", "succeeded", pids={"obj": "D"})]
    loop = [rec(0, "calibrated", "succeeded", pids={})] + lap * 5
    assert runner._stuck_reason(loop[:-1]) is not None and "10 of the last" in runner._stuck_reason(loop[:-1])
    assert runner._stuck_counted(rec(0, "registered", "succeeded")) is False
    assert runner._stuck_counted(rec(0, "held", "succeeded")) is True


def test_registrations_neither_count_nor_clear_a_standing_alert(tmp_path, scene):
    lap = [("held", "failed", "geometry_unmeasured", {"obj": "C", "grip": ["D"]}),
           ("supported_on", "failed", "geometry_unmeasured", {"obj": "C", "grip": ["D"], "ref": "B"}),
           ("registered", "succeeded", "registered", {"obj": "D"})]
    summary = smoke_with(tmp_path, scene, "repoint_loop", lap * 12, 40)
    alerts = summary["stuck_alerts"]
    assert alerts and alerts[0]["outcome_n"] == 14                 # the 10th counted failure (outcome 14)
    assert len(alerts) >= 2 and all(a.get("cleared_by_outcome_n") is None for a in alerts[:-1])


class ScriptedOutcomePolicy(runner.PlumbingHoldPolicy):
    """A runner-contract policy that reports a scripted outcome per act (the stuck-alert plumbing)."""

    script = ()

    def act(self, observation):
        decision = super().act(observation)
        if self.index < len(self.script):
            relation, status, reason, pids = self.script[self.index]
            decision.diagnostics["outcomes"] = [{"n": self.index + 1, "skill": relation, "relation": relation,
                                                 "status": status, "reason": reason, "pids": pids,
                                                 "step": self.index}]
        else:
            decision.diagnostics["outcomes"] = []
        return decision


def smoke_with(tmp_path, scene, name, script, steps):
    policy = type("P", (ScriptedOutcomePolicy,), {"script": tuple(script)})
    return runner.execute(tmp_path / name, config=CONFIG, sim_config=runner.build_sim_config("libero_10_t2")[0],
                          task_key="libero_10_t2", episode_index=0,
                          sim_factory=lambda cfg: world_sim(scene), video=False, smoke=True, smoke_steps=steps,
                          policy_factory=policy)


def test_stuck_alerts_fire_on_each_criterion(tmp_path, scene, capsys):
    summary = smoke_with(tmp_path, scene, "plan", [("plan", "rejected", "invalid_plan", {})], 3)
    assert [a["outcome_n"] for a in summary["stuck_alerts"]] == [1]
    assert "invalid_plan" in summary["stuck_alerts"][0]["reason"] and summary["stuck_alerts"][0]["automatic_cap"] is False
    assert "STUCK ALERT" in capsys.readouterr().err
    held = [("held", "succeeded", "hold_pass", {"obj": "C"})] + [("held", "failed", "hold_lost", {"obj": "C"})] * 6
    summary = smoke_with(tmp_path, scene, "held", held, 9)
    assert [a["outcome_n"] for a in summary["stuck_alerts"]] == [7]
    assert "re-establish 6 times" in summary["stuck_alerts"][0]["reason"]
    window = [("turned", "failed", "turn_slip", {"obj": f"A{i}"}) for i in range(10)]
    summary = smoke_with(tmp_path, scene, "window", window, 12)
    assert [a["outcome_n"] for a in summary["stuck_alerts"]] == [10]
    assert summary["stuck_alerts"][0]["reason"].startswith("10 of the last 10")
    assert summary["status"] == "diagnostic_complete"                            # alerts only, no cap


def test_a_standing_stuck_alert_repeats_while_the_failures_continue(tmp_path, scene, capsys):
    """Review F11(b): a repair loop with no success keeps alerting every STUCK_REPEAT_FAILURES failures (no cap)."""
    loop = [("turned", "failed", "precondition_unmet", {"obj": "A"})] * 34
    summary = smoke_with(tmp_path, scene, "loop", loop, 36)
    alerts = summary["stuck_alerts"]
    assert [a["outcome_n"] for a in alerts] == [10, 22, 34] and [a["repeat"] for a in alerts] == [0, 1, 2]
    assert all(a["reason"].startswith("still stuck: 12 further") for a in alerts[1:])
    assert [a["active"] for a in alerts] == [False, False, True]
    assert alerts[0]["superseded_by_outcome_n"] == 22 and alerts[1]["superseded_by_outcome_n"] == 34
    assert all(a["automatic_cap"] is False for a in alerts) and summary["status"] == "diagnostic_complete"
    assert capsys.readouterr().err.count("STUCK ALERT") == 3
    # A success clears the standing alert and resets the repeat count; the window rule applies again afterwards
    # (its window still holds the earlier failures), and the repeat rule then needs 12 further failures.
    cleared = loop[:10] + [("turned", "succeeded", "turned", {"obj": "A"})] + loop[:11]
    summary = smoke_with(tmp_path, scene, "cleared", cleared, 24)
    alerts = summary["stuck_alerts"]
    assert [a["outcome_n"] for a in alerts] == [10, 12] and [a["repeat"] for a in alerts] == [0, 0]
    assert alerts[0]["active"] is False and alerts[0]["cleared_by_outcome_n"] == 11
    assert alerts[1]["reason"].startswith("11 of the last 12") and alerts[1]["active"] is True


# ----------------------------------------------------------------------------------------------- 5. trace rows
def test_v2_event_rows_keep_the_runner_step_and_the_event_step(tmp_path, scene):
    class Emitting(runner.PlumbingHoldPolicy):
        def act(self, observation):
            decision = super().act(observation)
            decision.diagnostics["v2_events"] = [
                {"event": "step_start", "policy_step": self.index, "i": 0, "do": "turn", "obj": "A", "step": 99,
                 "mode": "x"},
                {"event": "observer", "policy_step": self.index, "key": "hold_floor", "verdict": "met"}]
            decision.diagnostics["pointing"] = [{"stage": "fine", "uv": [1.0, 2.0]}]
            return decision
    runner.execute(tmp_path / "ev", config=CONFIG, sim_config=runner.build_sim_config("libero_10_t2")[0],
                   task_key="libero_10_t2", episode_index=0, sim_factory=lambda cfg: world_sim(scene), video=False,
                   smoke=True, smoke_steps=3, policy_factory=Emitting)
    trace = rows(tmp_path / "ev" / "trace.jsonl")
    starts = [row for row in trace if row["event"] == "step_start"]
    assert [row["step"] for row in starts] == [0, 1, 2] and all(row["event_step"] == 99 for row in starts)
    assert all(row["mode"] == runner.SMOKE_MODE and row["policy_step"] == row["step"] for row in starts)
    assert all(row["event_mode"] == "x" for row in starts)                      # the payload mode is kept
    # Review F11(a): one copy of each v2 event (the action row carries only the count), and the act's pointing
    # results are "pointing_result" rows, never the v2 "pointing" event's name.
    actions = [row for row in trace if row["event"] == "action"]
    assert all("v2_events" not in row["diagnostics"] and row["v2_event_rows"] == 2 for row in actions)
    assert "v2_events" not in json.dumps(actions)
    assert [row["result"] for row in trace if row["event"] == "pointing_result"] == [{"stage": "fine",
                                                                                     "uv": [1.0, 2.0]}] * 3
    assert not any(row["event"] == "pointing" for row in trace)
    summary = json.loads((tmp_path / "ev" / "summary.json").read_text())
    assert summary["v2_event_counts"] == {"step_start": 3, "observer": 3}
    assert summary["runner_pointing_events"] == 3 and "pointing" in summary["nominal_events_missing"]


def test_plan_atom_check():
    key = {"goal_atoms": [["turnon", "flat_stove_1"], ["on", "moka_pot_1", "flat_stove_1_cook_region"]]}
    check = runner.plan_atom_check(key, TASK2_PLAN)
    assert [row["planned"] for row in check["atoms"]] == [True, True]
    assert [row["plan_steps"] for row in check["atoms"]] == [[0], [1]] and check["all_atoms_planned"] is True
    assert all(row["planned_with_text_match"] for row in check["atoms"])
    no_effect = json.loads(json.dumps(TASK2_PLAN))
    no_effect["steps"][0]["until"] = [{"o": "stop", "p": None, "txt": None}]
    no_effect["steps"] = no_effect["steps"][:1]
    check = runner.plan_atom_check(key, no_effect)
    assert [row["planned"] for row in check["atoms"]] == [False, False] and check["all_atoms_planned"] is False
    assert runner.plan_atom_check(key, None)["atoms"][0]["planned"] is None


def test_merge_plan_match_copies_planned_only_when_the_atoms_line_up():
    from gpt_field.agent_speedup_v2_truth import parse_check
    key = {"goal_atoms": [["turnon", "flat_stove_1"], ["on", "moka_pot_1", "flat_stove_1_cook_region"]]}
    merged = runner.merge_plan_match(parse_check(key, plan=TASK2_PLAN), runner.plan_atom_check(key, TASK2_PLAN))
    assert [row["planned"] for row in merged["atoms"]] == [True, True] and merged["planned_source"]
    assert merged["atoms"][0]["expected_relation"] == parse_check(key)["atoms"][0]["expected_relation"]
    merged = runner.merge_plan_match(parse_check(key), runner.plan_atom_check(key, None))
    assert [row["planned"] for row in merged["atoms"]] == [None, None] and merged["plan_available"] is False
    other = runner.plan_atom_check({"goal_atoms": key["goal_atoms"][:1]}, TASK2_PLAN)
    merged = runner.merge_plan_match(parse_check(key, plan=TASK2_PLAN), other)
    assert [row["planned"] for row in merged["atoms"]] == [None, None] and merged["planned_source"] is None


# ------------------------------------------------------------------------------------------- 6. preflight and CLI
def test_preflight_prints_real_v2_payloads():
    result = runner.preflight(CONFIG, runner.build_sim_config("libero_10_t2")[0], task_key="libero_10_t2")
    assert set(result["payloads"]) == {"plan", "point_fine", "point_coarse", "repair"}
    text = json.dumps(result["payloads"])
    assert "data:image/png;base64," not in text and "<data:image/png;base64" in text
    for name, payload in result["payloads"].items():
        assert payload["reasoning"] == {"effort": "medium"} and payload["store"] is False, name
        request = json.loads(payload["input"][0]["content"][0]["text"])
        assert request["task"] == T2.prompt and request["request"] == name, name
        assert len(result["payload_sha256"][name]) == 64
    assert result["agent_speedup_v2"]["instructions_sha256"] and result["agent_speedup_v2"]["plan_schema_sha256"]
    assert result["client"]["class"] == "AgentSpeedUpV2Client" and result["api_calls"] == 0
    assert "--live" in result["live_command"] and result["lint_check_require_all"]["require_all"] is True
    assert "[" not in result["live_command"] and "--store-dir" in result["live_command_options"]
    assert result["live_command_first_run"] == (result["live_command"] + " --stop-after-steps "
                                                f"{runner.LIVE_FIRST_RUN_STOP_AFTER_STEPS}")


@pytest.mark.parametrize("field", ["live_command", "live_command_first_run"])
def test_the_printed_live_commands_parse_as_printed(field, tmp_path, monkeypatch):
    """Review F9: the commands --preflight prints are copy-paste safe (no placeholder flags)."""
    import shlex
    command = runner.preflight(CONFIG, runner.build_sim_config("libero_10_t2")[0], task_key="libero_10_t2")[field]
    prefix, rest = command.split(" .venv/bin/python ", 1)
    assert prefix == f"cd {ROOT} && env -u CUDA_VISIBLE_DEVICES MUJOCO_EGL_DEVICE_ID=2"
    argv = shlex.split(rest.replace("$(date -u +%Y%m%dT%H%M%SZ)", "T"))
    assert argv[0] == runner.RUNNER_PATH and argv[argv.index("--output") + 1].startswith("runs/agent_speedup_v2_live_")
    argv[argv.index("--output") + 1] = str(tmp_path / "run")
    calls = []

    class FakeCredentials:
        def __init__(self, store_dir=None, profile=None):
            pass

        def access_token(self):
            raise AssertionError("never read in this test")
    import gpt_field.chatgpt_auth as auth
    monkeypatch.setattr(auth, "SubscriptionCredentials", FakeCredentials)
    monkeypatch.setattr(runner, "check_render_environment", lambda warn=None: {})
    monkeypatch.setattr(runner, "lint_check_summary", lambda root, require_all=True: {"passed": True})
    monkeypatch.setattr(runner, "execute", lambda output, **kwargs: calls.append(kwargs) or {
        "evaluation_completed": False, "diagnostic_complete": False})
    assert runner.main(argv[1:]) == 3 and calls[0]["smoke"] is False
    expected = runner.LIVE_FIRST_RUN_STOP_AFTER_STEPS if field == "live_command_first_run" else None
    assert calls[0]["stop_after_steps"] == expected


def test_a_broken_client_or_plan_build_refuses_a_live_run(tmp_path, scene, monkeypatch, capsys):
    """Review F6: a lint-clean but broken build (the 15:42 PlanParseError in preflight_payloads) refuses --live and
    execute(live) before any file is written; smokes are not affected."""
    from gpt_field import agent_speedup_v2_client as client_module
    from gpt_field.agent_speedup_v2_plan import PlanParseError

    def broken(task_prompt):
        raise PlanParseError("$.scene: length outside [1, 80]")
    monkeypatch.setattr(client_module, "preflight_payloads", broken)
    with pytest.raises(PlanParseError):
        runner.live_build_check(T2.prompt)
    with pytest.raises(ValueError, match="live build check fails"):
        live(tmp_path, world_sim(scene), stop_after_steps=2)
    assert not (tmp_path / "live").exists()
    monkeypatch.setattr(runner, "check_render_environment", lambda warn=None: {})
    monkeypatch.setattr(runner, "lint_check_summary", lambda root, require_all=True: {"passed": True})
    monkeypatch.setattr(runner, "execute", lambda *a, **k: pytest.fail("execute must not run"))
    with pytest.raises(SystemExit) as exit_info:
        runner.main(["--live", "--output", str(tmp_path / "run")])
    err = capsys.readouterr().err
    assert exit_info.value.code == 2 and "live build check fails (PlanParseError: $.scene" in err
    assert not (tmp_path / "run").exists()


def cli(*args):
    return subprocess.run([str(ROOT / ".venv/bin/python"), str(RUNNER_FILE), *args], capture_output=True, text=True,
                          cwd=ROOT, timeout=300)


def test_cli_mode_and_credential_refusals(tmp_path):
    existing = tmp_path / "exists"
    existing.mkdir()
    out = cli("--live", "--task", "libero_10_t2", "--output", str(existing))
    assert out.returncode == 2 and "output already exists" in out.stderr
    out = cli("--sim-smoke", "--output", str(tmp_path / "x"), "--store-dir", str(tmp_path))
    assert out.returncode == 2 and "only --live reads credentials" in out.stderr
    out = cli("--sim-smoke", "--output", str(tmp_path / "x"), "--stop-after-steps", "5")
    assert out.returncode == 2 and "--stop-after-steps requires --live" in out.stderr
    out = cli("--live", "--sim-smoke", "--output", str(tmp_path / "x"))
    assert out.returncode == 2 and "not allowed with argument" in out.stderr
    assert not (tmp_path / "x").exists()


def test_cli_live_path_builds_credentials_and_passes_the_live_arguments(tmp_path, monkeypatch):
    calls, made = [], []

    class FakeCredentials:
        def __init__(self, store_dir=None, profile=None):
            made.append((store_dir, profile))

        def access_token(self):
            raise AssertionError("never read in this test")
    import gpt_field.chatgpt_auth as auth
    monkeypatch.setattr(auth, "SubscriptionCredentials", FakeCredentials)
    monkeypatch.setattr(runner, "check_render_environment", lambda warn=None: {})
    monkeypatch.setattr(runner, "lint_check_summary", lambda root, require_all=True: {"passed": True})
    monkeypatch.setattr(runner, "execute", lambda output, **kwargs: calls.append((output, kwargs)) or {
        "evaluation_completed": False, "diagnostic_complete": False})
    code = runner.main(["--live", "--task", "libero_10_t2", "--output", str(tmp_path / "run"), "--store-dir",
                        str(tmp_path / "store"), "--profile", "me", "--stop-after-steps", "250"])
    assert code == 3 and made == [(tmp_path / "store", "me")]
    (output, kwargs), = calls
    assert output == tmp_path / "run" and kwargs["smoke"] is False and kwargs["stop_after_steps"] == 250
    assert kwargs["token_provider"].__self__.__class__ is FakeCredentials
    assert "oracle" not in kwargs["monitor_options"] and kwargs["episode_index"] == 0


def test_cli_live_refuses_a_failing_lint(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(runner, "check_render_environment", lambda warn=None: {})
    monkeypatch.setattr(runner, "lint_check_summary", lambda root, require_all=True: {
        "passed": False, "missing": ["agent_speedup_v2_relations.py"], "failing": {}})
    monkeypatch.setattr(runner, "execute", lambda *a, **k: pytest.fail("execute must not run"))
    with pytest.raises(SystemExit) as exit_info:
        runner.main(["--live", "--output", str(tmp_path / "run")])
    assert exit_info.value.code == 2 and "T0 lint fails" in capsys.readouterr().err


def test_cli_relation_smoke_default_steps(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(runner, "check_render_environment", lambda warn=None: {})
    monkeypatch.setattr(runner, "execute", lambda output, **kwargs: calls.append(kwargs) or {
        "evaluation_completed": False, "diagnostic_complete": True})
    assert runner.main(["--sim-smoke", "--smoke-script", "relation", "--output", str(tmp_path / "r")]) == 0
    assert calls[0]["smoke_steps"] == runner.DEFAULT_RELATION_SMOKE_STEPS and calls[0]["smoke_script"] == "relation"
    assert runner.main(["--sim-smoke", "--output", str(tmp_path / "p")]) == 0
    assert calls[1]["smoke_steps"] == runner.DEFAULT_SMOKE_STEPS and calls[1]["smoke_script"] == "plumbing"


# --------------------------------------------------------------------------------------------- 7. the v2 video
class FrameSink:
    """Stand-in for the session MP4 stream: keeps the rendered frames."""

    def __init__(self, path, *, fps=20):
        self.path, self.fps, self.frame_count, self.frames, self.closed = Path(path), fps, 0, [], False

    def write(self, frame, count=1):
        self.frames.append(frame)
        self.frame_count += count

    def close(self):
        self.closed = True


def recorder(tmp_path, **kwargs):
    from gpt_field.agent_speedup_v2_video import AgentSpeedUpV2VideoRecorder

    class Recorder(AgentSpeedUpV2VideoRecorder):
        writer_class = FrameSink
    return Recorder(tmp_path / "episode.mp4", episode_id="TEST / AGENT SPEEDUP V2", **kwargs)


def test_video_recorder_draws_plan_steps_observers_and_parts(tmp_path, scene):
    world = fakes.FakeTwoViewSim(scene)
    client = AgentSpeedUpV2Client(CONFIG, task_prompt=T2.prompt, transport=ScriptedV2Transport(answers()))
    policy = AgentSpeedUpV2Policy(client, relations=StandIn())
    video = recorder(tmp_path, offline_smoke=True)
    observation = world.observation()
    video.observe(observation, step=0)
    for _ in range(14):
        decision = policy.act(observation)
        decision.diagnostics["parts_uv"] = {"A": {"wrist": [250.0, 260.0], "external": [300.0, 200.0]},
                                            "B": {"wrist": None, "external": [100.0, 120.0]}}
        decision.diagnostics["observers"] = {"hold_floor": {"verdict": "met"}, "keepout:E": {"verdict": "violated"}}
        video.decision(observation, decision, decision_latency_s=0.01)
        after = world.step(decision.action)
        feedback = policy.observe_transition(observation, decision, after)
        video.executed(after, step=decision.diagnostics["step"], success=False, sim_status={}, feedback=feedback)
        observation = after
    video.close(status="diagnostic_complete")
    assert video.render_errors == 0, video.last_render_error
    assert video.plan == TASK2_PLAN and video.plan_status[0][0] == "done" and video.plan_status[1][0] == "done"
    assert video.model_calls >= 2 and video.summary()["plan_steps"] == 2
    assert {kind for kind, *_ in video._markers("wrist")} == {"part", "point"}
    frame = video.writer.frames[-1]
    assert frame.size == (1600, 960)
    index = [json.loads(line) for line in (tmp_path / "episode.events.jsonl").read_text().splitlines()]
    assert index[-1]["event"] == "terminal" and index[-1]["policy"] == "agent_speedup_v2"
    assert index[-1]["plan_status"]["0"][0] == "done" and "observers" in index[-1]


def test_video_recorder_survives_malformed_diagnostics(tmp_path, scene):
    from gpt_field.policy import Decision
    world = fakes.FakeTwoViewSim(scene)
    video = recorder(tmp_path)
    observation = world.observation()
    video.observe(observation, step=0)
    bad = {"step": 0, "controller": "agent_relation", "parts_uv": {"A": {"wrist": ["x", None]}, "B": 7},
           "observers": {"k": 3}, "relation": "not a dict", "v2_events": [{"event": "relation_plan", "plan": 5},
                                                                         {"event": "step_end", "i": "0"}, "junk"],
           "pointing": [{"camera": "wrist", "uv_upright": [float("nan"), 1.0]}, "x"], "outcome": "odd"}
    video.decision(observation, Decision((0.0,) * 6 + (-1.0,), bad), decision_latency_s=None)
    video.executed(world.step((0.0,) * 6 + (-1.0,)), step=0, success=False)
    video.close(status="user_stop")
    assert video.writer.closed and video.render_errors == 0, video.last_render_error


def test_video_module_strings_hold_no_vocabulary_token_and_no_control_module_imports_it():
    vocabulary = th.lint_vocabulary(LIBERO)["vocabulary"]
    hits = set()
    for node in ast.walk(ast.parse(VIDEO_FILE.read_text())):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            for token in th.tokens(node.value):
                hits |= set(th.token_hits(token, vocabulary, th.LINT_ALLOWLIST))
    assert hits == set()
    assert "agent_speedup_v2_video.py" not in th.CONTROL_MODULES + th.MODEL_INTERFACE_MODULES
    for name in th.CONTROL_MODULES + th.PRESENT_INTERFACE_MODULES:
        assert "agent_speedup_v2_video" not in (th.PACKAGE_DIR / name).read_text(), name


def test_live_run_with_video_writes_the_v2_event_index(tmp_path, scene, monkeypatch):
    from gpt_field.agent_speedup_v2_video import AgentSpeedUpV2VideoRecorder

    class Recorder(AgentSpeedUpV2VideoRecorder):
        writer_class = FrameSink
    sim = world_sim(scene, images=True)
    summary, output = live(tmp_path, sim, stop_after_steps=8, recorder_factory=Recorder)
    assert summary["status"] == "user_stop" and summary["video_error"] is None, summary["video_error"]
    index = [json.loads(line) for line in (output / "episode.events.jsonl").read_text().splitlines()]
    assert {row["event"] for row in index} >= {"observation", "skill_decision", "agent_action", "executed",
                                               "terminal"}
    assert all(row["policy"] == "agent_speedup_v2" and row["render_errors"] == 0 for row in index)
    assert np.isfinite(summary["timing_components"]["video_render_s"])
