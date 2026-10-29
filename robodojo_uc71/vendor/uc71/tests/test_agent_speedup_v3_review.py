"""Single-use approval provenance; all fixtures are offline and simulator-free."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest

from gpt_field.agent_speedup_v3_review import (
    DELEGATED_REVIEW, MANUAL_REVIEW, VERSION, ReviewError, ReviewStore,
    approval_metadata, delegated_review_preparation,
)

ROOT = Path(__file__).resolve().parents[1]
PLAN = {
    "scene": "One cup on the table.",
    "parts": [{"id": "A", "text": "cup: body", "move": "free", "w": 20, "x": 30}],
    "hold_true": [],
    "steps": [{"do": "hold", "obj": "A", "grip": [], "ref": None,
               "sense": "any", "keep": [], "until": []}],
}
NOTE = "User explicitly delegates review and approval of the one t1 retry initial plan to Codex."


def make_store(path, preparation=None):
    return ReviewStore.create(
        path, task_key="libero_10_t1", episode_index=0, task_prompt="test task",
        initial_observation={"state": [0]}, plan=deepcopy(PLAN),
        runtime_sha256={"policy.py": "a" * 64, "@configuration": "b" * 64},
        preparation=preparation)


def load_runner():
    spec = importlib.util.spec_from_file_location("speedup_v3_review_test_runner", ROOT / "scripts/evaluate_agent_speedup_v3.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_manual_default_stays_manual_and_new_version_requires_fresh_review(tmp_path):
    record = make_store(tmp_path / "review").load()
    assert record["version"] == VERSION == "agent-speedup-v3-11"
    assert record["status"] == "draft" and record["approved"] is None
    meta = approval_metadata(record["preparation"])
    assert meta["approval_mode"] == MANUAL_REVIEW
    assert meta["manual_review"] is True and meta["reviewer"] == "user"
    assert meta["automatically_approved"] is False


def test_delegation_requires_explicit_note_and_never_approves_while_preparing(tmp_path):
    for note in (None, "", "  ", 12, "x" * 4097):
        with pytest.raises(ReviewError):
            delegated_review_preparation(note)
    with pytest.raises(ReviewError):
        make_store(tmp_path / "automatic", {"approval_mode": "user_authorized_automatic", "manual_review": False})
    with pytest.raises(ReviewError):
        make_store(tmp_path / "missing-note", {"approval_mode": DELEGATED_REVIEW, "manual_review": False})
    record = make_store(tmp_path / "delegated", delegated_review_preparation(NOTE)).load()
    assert record["status"] == "draft" and record["approved"] is None
    assert record["reviewed_steps"] == []
    meta = approval_metadata(record["preparation"])
    assert meta["manual_review"] is False and meta["reviewer"] == "codex"
    assert meta["authorized_by"] == "user" and meta["authorization_note"] == NOTE
    assert meta["automatically_approved"] is False


def test_delegated_claim_preserves_step_reset_runtime_and_single_use_binding(tmp_path):
    store = make_store(tmp_path / "review", delegated_review_preparation(NOTE))
    record = store.load()
    claim = dict(output=tmp_path / "run", task_key=record["task_key"], episode_index=0,
                 initial_observation_sha256=record["initial_observation_sha256"],
                 runtime_sha256=record["runtime_sha256"])
    with pytest.raises(ReviewError):
        store.claim(**claim)
    with pytest.raises(ReviewError):
        store.approve(expected_revision=record["revision"], reviewed_steps=[])
    store.approve(expected_revision=record["revision"], reviewed_steps=[0])
    for changes in ({"episode_index": 1}, {"task_key": "libero_10_t2"},
                    {"initial_observation_sha256": "c" * 64},
                    {"runtime_sha256": {"policy.py": "a" * 64, "@configuration": "c" * 64}}):
        with pytest.raises(ReviewError):
            store.claim(**{**claim, **changes})
    assert store.claim(**claim)["status"] == "consumed"
    with pytest.raises(ReviewError):
        store.claim(**claim)


def test_delegation_note_is_part_of_immutable_review_context(tmp_path):
    store = make_store(tmp_path / "review", delegated_review_preparation(NOTE))
    record = json.loads(store.review_path.read_text())
    record["preparation"]["approval_provenance"]["authorization_note"] = "A different authorization"
    store.review_path.write_text(json.dumps(record))
    with pytest.raises(ReviewError, match="integrity"):
        store.load()


def test_delegation_changes_bound_config_and_active_identity_without_manual_claim(tmp_path):
    runner = load_runner()
    config = runner.base.configuration()
    sim_config, monitor = runner.base.build_sim_config("libero_10_t1")
    delegation = delegated_review_preparation(NOTE)
    manual = runner.execution_options(config, sim_config, monitor_options=monitor)
    delegated = runner.execution_options(config, sim_config, monitor_options=monitor, approval_preparation=delegation)
    assert manual != delegated
    assert manual["user_check"]["framework"]["approval_mode"] == MANUAL_REVIEW
    assert delegated["user_check"]["framework"]["approval_mode"] == DELEGATED_REVIEW
    module = runner._runtime(delegation)
    factory = module.reviewed_recorder_factory(lambda *a, **k: object(), object(), output_dir=tmp_path)
    factory(tmp_path / "episode.mp4", episode_id="AGENT SPEEDUP V2")
    active = json.loads((tmp_path / "active_agent.json").read_text())
    assert active["human_reviewed_initial_plan"] is False
    assert active["approved_by"] == "codex"
    assert active["approval_provenance"]["authorization_note"] == NOTE


def test_normalized_artifacts_report_agent_review_and_preserve_raw_records(tmp_path):
    runner = load_runner()
    for name in ("run_state.json", "task_evaluation.json", "agent_stats.json"):
        (tmp_path / name).write_text(json.dumps({"user_check": {"manual_review": True}}))
    (tmp_path / "manifest.json").write_text(json.dumps({"configuration": {"active_agent": {}}}))
    raw = tmp_path / "base_v2_artifacts"
    raw.mkdir()
    (raw / "summary.json").write_text('{"raw": true}\n')
    summary = {"user_check": {"manual_review": True}, "human_reviewed_plan": True}
    meta = approval_metadata(delegated_review_preparation(NOTE))
    result = runner._publish_approval_provenance(tmp_path, summary, meta)
    assert result["human_reviewed_plan"] is False
    assert result["user_check"]["approval_mode"] == DELEGATED_REVIEW
    for name in ("summary.json", "runner_summary.json", "run_state.json", "task_evaluation.json", "agent_stats.json"):
        data = json.loads((tmp_path / name).read_text())
        assert data["human_reviewed_plan"] is False and data["approved_by"] == "codex"
        assert data["user_check"]["manual_review"] is False
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["configuration"]["human_reviewed_plan"] is False
    assert manifest["identity_hash"] == runner.base.digest(manifest["configuration"])
    assert (raw / "summary.json").read_text() == '{"raw": true}\n'


def test_delegation_argument_is_prepare_only_and_no_autoapprove_flag_exists(tmp_path):
    runner = load_runner()
    for command in ("run", "start"):
        with pytest.raises(SystemExit) as caught:
            runner.main([command, "--review-dir", str(tmp_path / "review"),
                         "--output", str(tmp_path / "run"), "--delegated-review-note", NOTE])
        assert caught.value.code == 2


def test_evidence_acquisition_stop_is_not_an_infrastructure_or_success_result(tmp_path):
    runner = load_runner()
    for name in ("run_state.json", "task_evaluation.json", "agent_stats.json"):
        (tmp_path / name).write_text(json.dumps({"status": "infrastructure_error"}))
    summary = {"user_check": {}, "status": "infrastructure_error",
               "error": {"type": "EvidenceAcquisitionFailed", "message": "No relevant new evidence"}}
    result = runner._publish_approval_provenance(tmp_path, summary, approval_metadata())
    assert result["status"] == "evidence_acquisition_failed"
    assert not result["task_success"] and not result["completion_time_eligible"]
    for name in ("summary.json", "run_state.json", "task_evaluation.json"):
        record = json.loads((tmp_path / name).read_text())
        assert record["status"] == "evidence_acquisition_failed"
        assert not record["evaluation_completed"] and record["incomplete_episodes"] == 1


def test_preparation_and_execution_use_recovery_and_wait_is_reported(tmp_path):
    from gpt_field.agent_speedup_v3_client import V3SubscriptionAvailabilityRetryMixin
    runner = load_runner()
    runtime = runner._runtime()
    assert issubclass(runtime.UserCheckV2Client, V3SubscriptionAvailabilityRetryMixin)
    assert issubclass(runtime.AgentFirstV2Client, V3SubscriptionAvailabilityRetryMixin)
    summary = {"user_check": {}, "history_stats": {
        "transient_retry_requests": 2, "transient_retry_wait_s": 7.0,
        "transient_retry_codes": {"subscription_sharing_user_unavailable": 2},
        "overload_retry_requests": 0, "overload_retry_wait_s": 0.0}}
    result = runner._publish_approval_provenance(tmp_path, summary, approval_metadata())
    assert result["subscription_recovery"]["total_backoff_wait_s"] == 7.0
    assert result["model_calls"]["transient_retry_requests"] == 2


def test_low_overhead_runner_is_private_and_keeps_terminal_full_statistics():
    from gpt_field.agent_speedup_v3_video import AgentSpeedUpV3VideoRecorder
    runner = load_runner()
    original_execute = runner.base.execute
    original_recorder = runner.base._default_recorder_factory
    runtime = runner._runtime()
    assert runtime.base is not runner.base
    assert runner.base.execute is original_execute
    assert runner.base._default_recorder_factory is original_recorder
    assert runtime.base._v3_performance_adapted is True
    assert runtime.base._default_recorder_factory() is AgentSpeedUpV3VideoRecorder
    assert runtime.framework_settings()["performance"]["target_is_stop_condition"] is False


@pytest.mark.parametrize("call_ok", [True, False])
def test_terminal_act_timing_restores_measured_calls_once_without_false_acceptance(tmp_path, call_ok):
    from types import SimpleNamespace
    runner = load_runner()
    clock = SimpleNamespace(now=10.)
    class FakePolicy:
        def __init__(self):
            self._clock = lambda: clock.now
            self.step_index = 210
            self.reset()
        def reset(self):
            pass
        def _ask_agent(self, view, calls):
            clock.now += 2.
            calls.append(SimpleNamespace(kind="agent", latency_s=2., ok=call_ok,
                                         error=None if call_ok else "APIError"))
        def act(self, observation):
            calls = []
            self._ask_agent(None, calls)
            clock.now += .5
            raise RuntimeError("terminal policy application")
        def agent_stats(self):
            return {}
    policy = runner._timed_policy_class(FakePolicy)()
    with pytest.raises(RuntimeError):
        policy.act(None)
    stats = policy.agent_stats()
    assert stats["terminal_act_timings"][0]["decision_s"] == 2.5
    assert stats["terminal_act_timings"][0]["model_pipeline_s"] == 2.
    (tmp_path / "agent_stats.json").write_text(json.dumps(stats))
    summary = {"timing_components": {"decision_s": 12., "model_pipeline_s": 10.,
                                    "model_pipeline_s_by_kind": {"agent": 10.}},
               "model_calls": {"runner_calls_by_kind": {"agent": 24},
                               "accepted_by_kind": {"agent": 25 if call_ok else 24},
                               "failed_calls_by_kind": {}, "latency_s_by_kind": {"agent": 10.}}}
    result = runner._restore_terminal_act_accounting(tmp_path, summary)
    assert result["timing_components"]["decision_s"] == 14.5
    assert result["timing_components"]["model_pipeline_s"] == 12.
    assert result["timing_components"]["policy_compute_s"] == 2.5
    assert result["model_calls"]["runner_calls_by_kind"]["agent"] == 25
    assert result["model_calls"]["accepted_by_kind"]["agent"] == (25 if call_ok else 24)
    assert result["model_calls"]["failed_calls_by_kind"].get("agent", 0) == (0 if call_ok else 1)
    assert result["model_calls"]["latency_s_by_kind"]["agent"] == 12.
    again = runner._restore_terminal_act_accounting(tmp_path, result)
    assert again["timing_components"]["decision_s"] == 14.5
    assert (tmp_path / "terminal_policy_act_timing.json").exists()


def test_successful_policy_act_is_not_counted_again(tmp_path):
    runner = load_runner()
    class FakePolicy:
        def __init__(self):
            self._clock = lambda: 0.
            self.reset()
        def reset(self):
            pass
        def act(self, observation):
            return "normal decision"
        def agent_stats(self):
            return {}
    policy = runner._timed_policy_class(FakePolicy)()
    assert policy.act(None) == "normal decision"
    assert policy.agent_stats()["terminal_act_timings"] == []
    (tmp_path / "agent_stats.json").write_text(json.dumps(policy.agent_stats()))
    summary = {"timing_components": {"decision_s": 3.}}
    assert runner._restore_terminal_act_accounting(tmp_path, summary) == summary
    assert not (tmp_path / "terminal_policy_act_timing.json").exists()
