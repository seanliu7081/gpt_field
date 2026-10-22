"""Preparation and config binding for user-check; fake RGB and zero real calls."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest

from agent_speedup_v2_fakes import FakeV2Sim
from gpt_field.agent_speedup_v2_tasks import resolve_v2_task
from gpt_field.agent_speedup_v2_user_check import ReviewError


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "user_check_preparation_test_runner", ROOT / "scripts/evaluate_agent_speedup_v2_user_check.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)

PLAN = {
    "scene": "One cup on the table.",
    "parts": [{"id": "A", "text": "cup: body", "move": "free", "w": 20, "x": 30}],
    "hold_true": [],
    "steps": [{"do": "hold", "obj": "A", "grip": [], "ref": None,
               "sense": "any", "keep": [], "until": []}],
}
TASK = resolve_v2_task("libero_10_t2")


def settings():
    config = runner.base.configuration()
    sim_config, monitor = runner.base.build_sim_config(TASK.key)
    return config, sim_config, monitor


class DraftClient(runner.base.OfflinePlumbingClient):
    requests = []
    draft = PLAN

    def request_plan(self, external, wrist, **kwargs):
        self.requests.append({"external": external.shape, "wrist": wrist.shape, **kwargs})
        return copy.deepcopy(self.draft)


def prepare(tmp_path, *, expected=None, client=DraftClient, fake=None):
    DraftClient.requests = []
    config, sim_config, monitor = settings()
    sim = fake or FakeV2Sim(prompt=TASK.prompt)
    store = runner.prepare_review(
        tmp_path / "review", config=config, sim_config=sim_config, task_key=TASK.key,
        episode_index=0, monitor_options=monitor, client_factory=client,
        sim_factory=lambda _: sim, token_provider=lambda: "offline",
        expected_initial_observation_sha256=expected, video=False, stop_after_steps=10)
    return store, sim


def test_prepare_requests_one_draft_and_never_sends_policy_actions(tmp_path):
    store, fake = prepare(tmp_path)
    data = store.load()
    assert data["status"] == "draft"
    assert data["approved"] is None
    assert data["reviewed_steps"] == []
    assert data["plan"] == PLAN
    assert fake.actions == []
    assert fake.ops == ["start", "reset", "close"]
    assert len(DraftClient.requests) == 1
    assert DraftClient.requests[0]["external"] == (512, 512, 3)
    assert data["preparation"]["policy_actions"] == 0
    assert data["preparation"]["draft_model_s"] >= 0
    assert len(data["preparation"]["review_images"]) == 4
    for image in data["preparation"]["review_images"]:
        assert image["data_url"].startswith("data:image/png;base64,")
    frozen = store.path / "frozen_source"
    assert (frozen / runner.RUNNER_PATH).is_file()
    assert (frozen / runner.HTML_PATH).is_file()
    assert (frozen / "src/gpt_field/agent_speedup_v2.py").is_file()


def test_default_preparation_client_sends_high_payload(tmp_path):
    from gpt_field.agent_speedup_v2_client import FixedAnswers, ScriptedV2Transport

    config, sim_config, monitor = settings()
    sim = FakeV2Sim(prompt=TASK.prompt)
    transport = ScriptedV2Transport(FixedAnswers(plans=[PLAN]), effort="high")
    store = runner.prepare_review(
        tmp_path / "high_review", config=config, sim_config=sim_config,
        task_key=TASK.key, episode_index=0, monitor_options=monitor,
        sim_factory=lambda _: sim, token_provider=lambda: "offline",
        transport=transport, video=False)
    assert len(transport.calls) == 1
    assert transport.calls[0]["reasoning"]["effort"] == "high"
    assert store.load()["plan"] == PLAN
    assert sim.actions == []


def test_prepare_checks_reset_hash_before_any_model_call(tmp_path):
    fake = FakeV2Sim(prompt=TASK.prompt)
    with pytest.raises(ReviewError, match="no model call"):
        prepare(tmp_path, expected="0" * 64, fake=fake)
    assert DraftClient.requests == []
    assert fake.actions == []
    assert fake.closed
    assert not (tmp_path / "review/review.json").exists()


def test_prepare_does_not_hide_unsupported_model_draft(tmp_path):
    class UnsupportedDraft(DraftClient):
        draft = copy.deepcopy(PLAN)
        draft["steps"][0].update(do="slide", sense="open",
                                until=[{"o": "stop", "p": None, "txt": None}])
        draft["parts"][0]["move"] = "sliding"

    store, fake = prepare(tmp_path, client=UnsupportedDraft)
    assert not store.load()["validation"]["valid"]
    with pytest.raises(ReviewError, match="not implemented"):
        store.approve(expected_revision=0, reviewed_steps=[0])
    assert fake.actions == []


def test_preparation_requires_fresh_directory(tmp_path):
    (tmp_path / "review").mkdir()
    with pytest.raises(FileExistsError):
        prepare(tmp_path)
    assert DraftClient.requests == []


def test_configuration_change_requires_new_review(tmp_path):
    store, _ = prepare(tmp_path)
    store.approve(expected_revision=0, reviewed_steps=[0])
    config, sim_config, monitor = settings()
    with pytest.raises(ReviewError, match="Runtime code changed"):
        runner.execute_reviewed(
            tmp_path / "run", review_dir=store.path, config=config, sim_config=sim_config,
            task_key=TASK.key, episode_index=0, monitor_options=monitor,
            token_provider=lambda: "offline", video=False, stop_after_steps=11)
    assert store.load()["status"] == "approved"
    assert not (tmp_path / "run").exists()


def test_runtime_inventory_includes_new_sources_and_execution_configuration():
    config, sim_config, monitor = settings()
    inventory = runner.runtime_inventory(config, sim_config, monitor_options=monitor)
    assert runner.RUNNER_PATH in inventory
    assert runner.HTML_PATH in inventory
    assert "src/gpt_field/agent_speedup_v2_user_check_client.py" in inventory
    assert "src/gpt_field/agent_speedup_v2_user_check_control.py" in inventory
    for name in ("framework", "runtime", "skills", "motion"):
        assert f"src/gpt_field/agent_speedup_v2_user_check_{name}.py" in inventory
    assert "src/gpt_field/robot_facts_v2.json" in inventory
    other = runner.runtime_inventory(config, sim_config, monitor_options=monitor, video=False)
    assert inventory["@configuration"] != other["@configuration"]
    assert {k: v for k, v in inventory.items() if not k.startswith("@")} == {
        k: v for k, v in other.items() if not k.startswith("@")}


def test_preflight_only_advertises_mandatory_review_commands():
    config, sim_config, monitor = settings()
    result = runner.preflight(
        config, sim_config, task_key=TASK.key, episode_index=0, monitor_options=monitor)
    assert result["method"] == runner.METHOD
    assert result["api_calls"] == 0
    assert not result["simulator_started"]
    assert not result["credentials_checked"]
    assert "user_check.py start" in result["live_command"]
    assert "evaluate_agent_speedup_v2.py --live" not in result["live_command"]
    assert result["client"]["class"] == "ReviewedV2Client"
    assert result["client"]["reasoning_effort"] == "high"
    assert result["preparation_client"]["class"] == "UserCheckV2Client"
    assert result["preparation_payload"]["reasoning"]["effort"] == "high"
    assert result["reviewed_grounding_payload"]["reasoning"]["effort"] == "high"
    assert result["automatic_replan_payload"]["reasoning"]["effort"] == "high"
    assert result["base_v2_reference_payloads"]["plan"]["reasoning"]["effort"] == "medium"
    assert result["framework"]["local_motion"]["controller"]["max_steps"] == 5
    assert result["reviewed_grounding_payload_sha256"]
    assert set(result["client"]["effective_instructions_sha256"]) == {"plan", "replan", "repair", "review"}
    assert result["automatic_model_replanning"] is True
    assert result["approval_scope"] == "initial_plan_only"
    assert result["automatic_replan_payload_sha256"]
    assert result["lowering_control"] == runner.control_settings()
    assert runner.PLAN_CONFLICT_RULE in result["stopping_rule"]
    assert not any(rule.startswith("recovery_exhausted") for rule in result["stopping_rule"])


def test_execution_binding_records_initial_only_review_and_recovery_controller():
    config, sim_config, monitor = settings()
    options = runner.execution_options(config, sim_config, monitor_options=monitor)
    assert options["user_check"] == {
        "version": runner.VERSION, "approval_scope": "initial_plan_only",
        "automatic_model_replanning": True, "lowering_control": runner.control_settings(),
        "framework": runner.framework_settings(),
    }


def test_completed_artifacts_distinguish_initial_approval_from_model_revisions(tmp_path):
    from test_agent_speedup_v2_user_check_integration import approved_fake_run

    module, review, sim, kwargs = approved_fake_run(tmp_path)
    output = tmp_path / "recorded"
    summary = module.execute_reviewed(output, **kwargs)
    assert summary["status"] == "user_stop", summary.get("error")
    meta = summary["user_check"]
    assert meta["approval_scope"] == "initial_plan_only"
    assert meta["automatic_model_replanning"] is True
    assert meta["fixed_semantics"] is False
    assert meta["user_changes_during_test"] is False
    history = json.loads((output / meta["plan_history_file"]).read_text())
    assert history["plan_revision_count"] == meta["plan_revision_count"]
    assert history["initial_plan_grounded"] is True
    assert history["conflict"] is None
    assert json.loads((output / "approved_plan.json").read_text()) == review.load()["plan"]
    assert meta["lowering_control"] == module.control_settings()
    framework = json.loads((output / "user_check_framework.json").read_text())
    assert framework["task"]["task_prompt"] == TASK.prompt
    assert framework["version"] == module.FRAMEWORK_VERSION
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["configuration"]["user_check"] == meta
    for name in ("agent_speedup_v2_user_check.py", "agent_speedup_v2_user_check_client.py",
                 "agent_speedup_v2_user_check_control.py"):
        assert (output / "frozen_source" / name).read_bytes() == (ROOT / "src/gpt_field" / name).read_bytes()


@pytest.mark.parametrize("conflict", [None, {"reason": "A different step is required"}])
def test_injected_recorder_keeps_variant_label_and_true_terminal_status(conflict):
    captured = {}

    class FakeRecorder:
        def __init__(self, path, **kwargs):
            captured.update(path=path, **kwargs)

        def observe(self, value, **kwargs):
            captured["observed"] = value

        def close(self, **kwargs):
            captured["closed"] = kwargs
            return "closed"

    lock = runner.PlanLock(PLAN)
    factory = runner.reviewed_recorder_factory(FakeRecorder, lock)
    recorder = factory("fake.mp4", episode_id="TEST / AGENT SPEEDUP V2 / NO MODEL CALLS",
                       offline_smoke=True)
    recorder.observe("fake RGB", step=0)
    lock.conflict = conflict
    assert recorder.close(status="user_stop", error="base interruption") == "closed"
    assert captured["episode_id"] == "TEST / SPEED UP V2 USER CHECK / NO MODEL CALLS"
    assert captured["offline_smoke"] is True
    assert captured["observed"] == "fake RGB"
    if conflict is None:
        assert captured["closed"] == {"status": "user_stop", "error": "base interruption"}
    else:
        assert captured["closed"]["status"] == "plan_conflict"
        assert json.loads(captured["closed"]["error"]) == {
            "type": "PlanConflict", "details": conflict}
