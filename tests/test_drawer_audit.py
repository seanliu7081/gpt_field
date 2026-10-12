"""Recorded drawer-task replay with mock transport and mock simulator only."""

from dataclasses import replace
import json

import pytest

from gpt_field.distance_tasks import resolve_distance_task
from gpt_field.experiment import digest
from test_distance_audit import auditor, rewrite, rewrite_trace, skip_fake_video
from test_distance_policy import intent
from test_distance_runner import SETTINGS
from test_field_history import FakeTransport, envelope
from test_field_history_runner import FakeRecorder, fake_simulator, observation, rows
from test_hybrid_subscription_runner import runner, sim_config


def drawer_fixture(tmp_path, observation):
    task = resolve_distance_task("drawer_bowl")
    obs = replace(observation, prompt=task.prompt)
    base, _ = fake_simulator(obs, success_at=11)

    class DrawerSimulator(base):
        def list_tasks(self):
            return [{"task_id": i, "prompt": task.prompt if i == task.task_id else "unused fixture task"}
                    for i in range(task.task_id + 1)]

        def reset(self, task_id, episode_index):
            assert self.config.suite == task.suite
            assert (task_id, episode_index) == (task.task_id, 0)
            return super().reset(0, 0)

    approach = intent()
    approach["target_region"] = "drawer_handle"
    opening = intent(direct=True, phase="drawer_open")
    opening["target_region"] = "drawer_handle"
    verify = intent(hold=True, phase="verify")
    verify["target_region"] = "drawer_interior"
    values = (approach, opening, verify)
    transport = FakeTransport(*(envelope(raw, id=f"drawer-audit-{i}", model="gpt-6-astra",
                                        reasoning={"effort": "medium"}) for i, raw in enumerate(values)))
    run = tmp_path / "drawer"
    summary = runner.execute_episode(run, config=runner.configuration(), sim_config=sim_config(suite=task.suite),
        token_provider=lambda: "fixture-oauth", transport=transport, sim_factory=DrawerSimulator,
        recorder_factory=FakeRecorder, task_key=task.key, **SETTINGS)
    assert summary["status"] == "success", summary.get("error")
    return run


def test_drawer_task_full_replay_uses_correct_prompt_suite_and_no_upward_recovery(
        tmp_path, observation, skip_fake_video):
    run = drawer_fixture(tmp_path, observation)
    report = auditor.audit_run(run, tmp_path / "audit")
    assert report["passed"], report["issues"]
    assert report["distance_task"] == "drawer_bowl"
    assert (report["suite"], report["task_id"], report["task"]) == (
        "libero_goal", 3, "open the top drawer and put the bowl inside")
    assert report["executed_actions"] == 11 and report["replayed_recorded_requests"] == 3
    assert report["distance_event_reasons"] == {
        "goal_reached": 1, "field_budget_exhausted": 1, "placement_unverified": 1}
    assert report["synthetic_test_fixture"] and not report["real_task_completion_verified"]
    trace = rows(run / "trace.jsonl")
    actions = [r for r in trace if r["event"] == "action"]
    assert all(not r["diagnostics"]["local_recovery_active"] for r in actions)
    assert all(r["action"][:3] == [0., 0., 0.] for r in actions[6:])


@pytest.mark.parametrize("target", ["task_index", "suite", "recovery_rule", "observed_prompt", "model_region"])
def test_drawer_audit_rejects_task_contract_tampering(tmp_path, observation, skip_fake_video, target):
    run = drawer_fixture(tmp_path, observation)
    if target in ("task_index", "suite", "recovery_rule"):
        manifest = json.loads((run / "manifest.json").read_text())
        config = manifest["configuration"]
        if target == "task_index":
            config["task_id"] = 0
        elif target == "suite":
            config["simulator"]["suite"] = "libero_spatial"
        else:
            config["task_spec"]["allow_upward_reobservation"] = True
        manifest["identity_hash"] = digest(config)
        rewrite(run / "manifest.json", manifest)
    elif target == "observed_prompt":
        trace = rows(run / "trace.jsonl")
        trace[0]["observation"]["prompt"] = resolve_distance_task("bowl_plate").prompt
        rewrite_trace(run, trace)
    else:
        path = sorted((run / "requests").glob("*.response.json"))[0]
        response = json.loads(path.read_text())
        message = next(item for item in response["output"] if item["type"] == "message")
        content = next(part for part in message["content"] if part["type"] == "output_text")
        raw = json.loads(content["text"])
        raw["target_region"] = "plate"
        content["text"] = json.dumps(raw)
        rewrite(path, response)
    report = auditor.audit_run(run, tmp_path / "audit")
    assert not report["passed"]
    expected = {"task_index": "requested task language, task index", "suite": "recorded task suite",
                "recovery_rule": "task registry and recovery restrictions", "observed_prompt": "exact selected task prompt",
                "model_region": "shared replay raised"}[target]
    assert any(expected in issue for issue in report["issues"]), report["issues"]


def test_unknown_task_cannot_be_silently_audited_as_the_bowl_task(tmp_path, observation, skip_fake_video):
    run = drawer_fixture(tmp_path, observation)
    manifest = json.loads((run / "manifest.json").read_text())
    manifest["configuration"]["distance_task"] = "unregistered_task"
    rewrite(run / "manifest.json", manifest)
    with pytest.raises(ValueError):
        auditor.audit_run(run, tmp_path / "audit")
    assert not (tmp_path / "audit").exists()
