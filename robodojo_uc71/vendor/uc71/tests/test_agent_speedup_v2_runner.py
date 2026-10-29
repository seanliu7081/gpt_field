"""agent_SpeedUp_v2 runner skeleton (M0) with fake simulators only; zero model calls."""

import importlib.util
import json
from pathlib import Path
import subprocess

import pytest

from agent_speedup_v2_fakes import FakeV2Sim, generic_record, truth_result, v1_record, v2_status
from gpt_field.direct_history import TASK as V1_BOWL_TASK
from gpt_field.agent_speedup_v2_tasks import V2_TASK_KEYS, resolve_v2_task

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("evaluate_agent_speedup_v2", ROOT / "scripts/evaluate_agent_speedup_v2.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
CONFIG = runner.configuration()
T2 = resolve_v2_task("libero_10_t2")
FINAL = ("summary.json", "runner_summary.json", "task_evaluation.json", "episodes.jsonl", "run_state.json",
         "ledger.json", "agent_stats.json", "manifest.json", "trace.jsonl", "initial_observation_check.json")


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def sim_config(key="libero_10_t2"):
    return runner.build_sim_config(key)[0]


def chained(key):
    return sim_config(key).collision_check_mode != "none"


def run(tmp_path, *, key="libero_10_t2", name="out", smoke_steps=12, sim=None, **kwargs):
    task = resolve_v2_task(key)
    fake = sim or FakeV2Sim(prompt=task.prompt, task_id=task.task_id, chained=chained(key))
    output = tmp_path / name
    summary = runner.execute(output, config=CONFIG, sim_config=sim_config(key), task_key=key, episode_index=0,
                             sim_factory=lambda cfg: fake, video=False, smoke=True, smoke_steps=smoke_steps,
                             expected_initial_observation_sha256=kwargs.pop("expected", None), **kwargs)
    return summary, output, fake


def test_plumbing_smoke_artifacts_and_protocol(tmp_path):
    summary, output, fake = run(tmp_path)
    assert summary["status"] == "diagnostic_complete" and summary["diagnostic_complete"]
    assert summary["steps"] == 12 and len(fake.actions) == 12
    for name in FINAL:
        assert (output / name).is_file(), name
    assert summary["api_calls"] == 0 and summary["evaluation_scored"] is False
    assert summary["task_success"] is False and summary["v2_success"] is False
    events = [row["event"] for row in rows(output / "trace.jsonl")]
    assert events[0] == "reset" and events[1] == "threshold_set"
    assert events.count("executed") == 12 and events.count("feedback") == 12
    # diagnosis-only truth: terminal, then the counterfactual, then the answer key, all before close
    assert fake.ops[-4:] == ["truth", "settle_truth", "answer_key", "close"]
    for name in ("physical_truth.json", "answer_key.json", "parse_check.json"):
        assert (output / name).is_file()
    assert summary["physical_truth"]["diagnosis_only"] and summary["physical_truth"]["affects_completion"] is False
    evaluation = json.loads((output / "task_evaluation.json").read_text())
    assert set(evaluation) == {"status", "task_success", "libero_success", "collision", "failure_reason",
                               "evaluation_completed", "success_rate", "scored_episodes", "incomplete_episodes",
                               "completion_time_eligible"}
    assert summary["timing_components"]["model_pipeline_s"] == 0.0
    assert summary["control_without_video_s"] is not None


def test_manifest_identity_and_frozen_source(tmp_path):
    _, output, _ = run(tmp_path)
    manifest = json.loads((output / "manifest.json").read_text())
    config = manifest["configuration"]
    assert manifest["identity_hash"]
    assert config["collision_policy"] == "record_only" and config["collision_monitor"] == "generic_contact_v2"
    assert config["simulator"]["seed"] == 7 and config["simulator"]["settle_steps"] == 10
    assert config["simulator"]["render_size"] == config["simulator"]["image_size"] == 512
    assert config["simulator"]["max_steps"] is None and config["simulator"]["stop_on_success"] is True
    assert config["simulator"]["include_arm_joints"] is False
    chain = {item["path"]: item["sha256"] for item in config["helper_chain"]}
    assert set(chain) == set(runner.HELPER_CHAIN) and all(chain.values())
    meta = config["agent_speedup_v2"]
    assert meta["constants_sha256"] and meta["lint"]["vocabulary_sha256"] and meta["lint"]["allowlist_sha256"]
    assert meta["lint"]["part_nouns_sha256"] and "handle" in meta["lint"]["part_nouns"]   # D2
    assert meta["rulings_assumed"]["R-1v_vertical_probe_moves_looming"] is True
    assert meta["rulings_assumed"]["R-3_evaluator_tuned_constants"] is False
    frozen = output / "frozen_source"
    for name in ("agent_speedup_v2_sim.py", "agent_speedup_v2_truth.py", "evaluate_agent_speedup_v2.py",
                 "evaluate_agent_speedup_v1.py", "evaluate_bowl_hybrid_subscription.py", "robot_facts_v2.json",
                 "libero.py"):
        assert (frozen / name).is_file(), name


def test_constants_and_metadata_identical_for_every_key():
    """AC-1(a): one constants register, one lint vocabulary and one instruction identity for every key."""
    first = runner.agent_metadata(runner.DEFAULT_LIBERO_ROOT)
    for key in V2_TASK_KEYS:
        resolve_v2_task(key)
        assert runner.agent_metadata(runner.DEFAULT_LIBERO_ROOT) == first
    from gpt_field.agent_speedup_v2_client import instructions_sha256   # RUN-EXT (BI): the v2 client's identity
    assert first["instructions_sha256"] == instructions_sha256()


@pytest.mark.parametrize("key", V2_TASK_KEYS)
def test_client_receives_only_the_official_reset_sentence(key, tmp_path):
    """AC-1(b) for every registry key: the client's only task input is the reset observation's sentence."""
    seen = []

    class RecordingClient(runner.OfflinePlumbingClient):
        def __init__(self, config=None, **kwargs):
            seen.append(kwargs)
            super().__init__(config, **kwargs)

    summary, output, _ = run(tmp_path, key=key, client_factory=RecordingClient, smoke_steps=1)
    assert summary["status"] == "diagnostic_complete", summary["error"]
    task = resolve_v2_task(key)
    assert len(seen) == 1 and set(seen[0]) == {"token_provider", "task_prompt", "transport", "log_dir"}
    assert seen[0]["task_prompt"] == task.prompt == summary["client_task_prompt"]
    if key != "bowl_plate":
        assert seen[0]["task_prompt"] != V1_BOWL_TASK
    identity = json.loads((output / "client.json").read_text())
    assert identity["task_prompt"] == task.prompt and identity["class"].endswith("RecordingClient")


def test_client_identity_recorded(tmp_path):
    """PF-8: v1-style client identity (class, model, prompt version, effort, history protocol, prompt)."""
    summary, output, _ = run(tmp_path, smoke_steps=1)
    identity = json.loads((output / "client.json").read_text())
    assert identity == {"class": "OfflinePlumbingClient", "model": CONFIG.model,
                        "prompt_version": runner.OfflinePlumbingClient.prompt_version, "reasoning_effort": None,
                        "history_protocol": "stateless_offline_plumbing",
                        "initial_history_settings": identity["initial_history_settings"], "task_prompt": T2.prompt}
    client = json.loads((output / "manifest.json").read_text())["configuration"]["client"]
    assert client["prompt_version"] == runner.OfflinePlumbingClient.prompt_version
    assert client["reasoning_effort"] is None and client["identity_file"] == "client.json"


def test_policy_actions_do_not_depend_on_the_task_sentence(tmp_path):
    """TLR-4 pattern (behavioural invariance; the v2 policy gets the same test in M6): the same scripted policy
    with two different task sentences gives identical action traces."""
    _, _, first = run(tmp_path, key="libero_10_t2", name="a", smoke_steps=40)
    _, _, second = run(tmp_path, key="libero_goal_g7", name="b", smoke_steps=40)
    assert resolve_v2_task("libero_10_t2").prompt != resolve_v2_task("libero_goal_g7").prompt
    assert first.actions == second.actions and len(first.actions) == 40


def test_client_requires_task_prompt_and_rejects_other_text():
    with pytest.raises(TypeError):
        runner.OfflinePlumbingClient(CONFIG)
    client = runner.OfflinePlumbingClient(CONFIG, task_prompt=T2.prompt)
    fake = FakeV2Sim(prompt=resolve_v2_task("drawer_bowl").prompt)
    with pytest.raises(ValueError):
        client.check_observation(fake.reset(3, 0))


def test_policy_gets_no_task_key(tmp_path):
    seen = []

    def factory(client, **kwargs):
        seen.append(kwargs)
        return runner.PlumbingHoldPolicy(client)
    run(tmp_path, policy_factory=factory, smoke_steps=2)
    assert seen == [{}]


def test_official_task_mismatch_fails_before_reset(tmp_path):
    fake = FakeV2Sim(prompt=T2.prompt, tasks=[{"task_id": 2, "prompt": "some other sentence"}])
    summary, output, _ = run(tmp_path, sim=fake)
    assert summary["status"] == "infrastructure_error" and "reset" not in fake.ops
    assert summary["physical_truth"]["verdict"] == "truth_unavailable"


def test_reset_hash_mismatch_fails_closed_before_any_action(tmp_path):
    summary, _, fake = run(tmp_path, expected="0" * 64)
    assert summary["status"] == "infrastructure_error" and fake.actions == []
    assert "truth" not in fake.ops


def test_missing_frozen_hash_fails_closed():
    with pytest.raises(ValueError, match="fail-closed"):
        runner.resolve_expected_sha("libero_10_t2", 7, None)
    assert runner.resolve_expected_sha("libero_10_t2", 0, None) == (T2.frozen_reset_sha256[0],
                                                                    "registry_frozen_reset")
    assert runner.resolve_expected_sha("libero_10_t2", 7, "a" * 64) == ("a" * 64, "argument")


def test_operator_stop_file_ends_at_a_boundary(tmp_path):
    class StopAfter(runner.PlumbingHoldPolicy):
        def observe_transition(self, before, decision, after):
            result = super().observe_transition(before, decision, after)
            if self.index == 3:
                (tmp_path / "out" / runner.OPERATOR_STOP_FILE).write_text("{}")
            return result
    summary, output, fake = run(tmp_path, policy_factory=StopAfter)
    assert summary["status"] == "user_stop" and summary["steps"] == 3
    assert any(row["event"] == "operator_stop" for row in rows(output / "trace.jsonl"))
    assert "truth" in fake.ops   # truth still reported for a user stop


def test_raw_success_ends_the_episode_and_truth_never_changes_it(tmp_path):
    held = truth_result(False, True, ["not_held", "at_rest"])
    settled = truth_result(True, True, [], state="after_release_settle")
    fake = FakeV2Sim(prompt=T2.prompt, success_at=5, terminal_truth=held, settled_truth=settled)
    summary, output, _ = run(tmp_path, sim=fake)
    assert summary["status"] == "diagnostic_sim_terminated" and summary["steps"] == 5
    assert summary["libero_success"] is True
    assert summary["physical_truth"]["verdict"] == "physical_success"
    assert summary["physical_truth"]["terminal_failure_is_hold_only"] is True
    perched = truth_result(False, True, ["lowest_point_below_rim", "not_perched_on_rim"])
    fake = FakeV2Sim(prompt=T2.prompt, success_at=5, terminal_truth=perched, settled_truth=perched)
    summary, _, _ = run(tmp_path, name="perch", sim=fake)
    assert summary["status"] == "diagnostic_sim_terminated"
    assert summary["physical_truth"]["verdict"] == "lenient_success" and summary["v2_success"] is False


def test_truth_failure_is_a_cleanup_error_not_an_outcome_change(tmp_path):
    fake = FakeV2Sim(prompt=T2.prompt, truth_error=RuntimeError("worker truth failed"))
    summary, output, _ = run(tmp_path, sim=fake)
    assert summary["status"] == "diagnostic_complete"
    assert any(item.get("stage") == "truth_terminal" for item in summary["cleanup_errors"])
    assert not (output / "physical_truth.json").exists()


class _QuotaAt3(runner.PlumbingHoldPolicy):
    def act(self, observation):
        if self.index == 3:
            raise runner.APIError("quota")
        return super().act(observation)


@pytest.mark.parametrize("truth_error", [None, RuntimeError("worker truth failed")])
def test_truth_failure_never_promotes_a_subscription_error(truth_error, tmp_path):
    """PF-2: a truth failure stays a cleanup_error; the status stays subscription_error."""
    fake = FakeV2Sim(prompt=T2.prompt, truth_error=truth_error)
    summary, _, _ = run(tmp_path, sim=fake, policy_factory=_QuotaAt3)
    assert summary["status"] == "subscription_error" and summary["steps"] == 3
    stages = [item.get("stage") for item in summary["cleanup_errors"]]
    assert stages == (["truth_terminal"] if truth_error else [])


def test_interrupt_inside_a_simulator_request_skips_truth(tmp_path):
    """PF-1: an interrupt inside sim.step leaves the step reply in the pipe; reading truth would read it."""
    fake = FakeV2Sim(prompt=T2.prompt, interrupt_at=5)
    summary, output, _ = run(tmp_path, sim=fake)
    assert summary["status"] == "user_stop" and summary["steps"] == 4 and summary["pending_action_index"] == 4
    assert not {"truth", "settle_truth", "answer_key"} & set(fake.ops) and fake.ops[-1] == "close"
    assert summary["physical_truth"]["verdict"] == "truth_unavailable"
    assert "reply in flight" in summary["physical_truth"]["skipped"]
    assert not (output / "physical_truth.json").exists() and not (output / "answer_key.json").exists()
    assert summary["cleanup_errors"] == []


def test_misaligned_truth_replies_are_rejected(tmp_path):
    """PF-1: every truth reply must have the shape of the op sent; a stale reply is a cleanup_error."""
    stale = {"observation": {"observation/state": [0.0] * 8}, "status": v2_status()}
    fake = FakeV2Sim(prompt=T2.prompt, stale_reply=stale)
    summary, output, _ = run(tmp_path, sim=fake)
    assert summary["status"] == "diagnostic_complete"
    errors = {item["stage"]: item["type"] for item in summary["cleanup_errors"]}
    assert errors == {"truth_terminal": "TruthReplyShapeError", "answer_key": "TruthReplyShapeError"}
    assert "settle_truth" not in fake.ops     # no counterfactual without a trusted terminal truth
    assert summary["physical_truth"]["verdict"] == "truth_unavailable"
    assert not (output / "physical_truth.json").exists() and not (output / "answer_key.json").exists()


def test_truth_reply_shapes():
    good = truth_result(True, True)
    assert runner._truth_reply_shape_ok("truth_terminal", good)
    assert not runner._truth_reply_shape_ok("truth_after_release_settle", good)
    assert runner._truth_reply_shape_ok("truth_after_release_settle", truth_result(True, True,
                                                                                    state="after_release_settle"))
    assert runner._truth_reply_shape_ok("answer_key", {"goal_atoms": []})
    for label in runner.TRUTH_READS:
        assert not runner._truth_reply_shape_ok(label, {"observation": {}, "status": {}})
        assert not runner._truth_reply_shape_ok(label, None)


def test_calibration_closes_are_not_the_first_close(tmp_path):
    """PF-7: closes marked diagnostics["calibration_probe"] are skipped by first_close_event."""
    class CalibratingPolicy(runner.PlumbingHoldPolicy):
        def act(self, observation):
            decision = super().act(observation)
            if self.index < 15:          # steps 10..14 are calibration closes
                decision.diagnostics["calibration_probe"] = True
            return decision
    summary, _, _ = run(tmp_path, smoke_steps=30, policy_factory=CalibratingPolicy)
    assert summary["first_close_event"]["action_index"] == 15
    summary, _, _ = run(tmp_path, name="plain", smoke_steps=30)
    assert summary["first_close_event"]["action_index"] == 10


def test_recorded_collisions_are_record_only_and_per_class(tmp_path):
    fake = FakeV2Sim(prompt=T2.prompt, collision_at=4)
    summary, output, _ = run(tmp_path, sim=fake)
    assert summary["status"] == "diagnostic_complete" and summary["steps"] == 12
    events = [row for row in rows(output / "trace.jsonl") if row["event"] == "recorded_collision"]
    assert [e["class"] for e in events] == ["robot_scene"] and events[0]["affects_completion"] is False
    assert events[0]["step"] == 3 and events[0]["first_detection"] is True


def test_chained_v1_record_invariants(tmp_path):
    """PF-6: the chained v1 record must be present, enabled, task_manipulation_v1 and record-only."""
    assert runner._collision_v1_record(v2_status(v1=v1_record()), True)["mode"] == "task_manipulation_v1"
    assert runner._collision_v1_record(v2_status(), False) is None
    with pytest.raises(ValueError):
        runner._collision_v1_record(v2_status(v1=v1_record()), False)
    for bad in (None, {**v1_record(), "enabled": False}, {**v1_record(), "mode": "bowl_obstacles_v1"},
                {**v1_record(), "stop_on_collision": True}, {**v1_record(), "collision_policy": "stop"}):
        with pytest.raises(ValueError, match="record-only"):
            runner._collision_v1_record(v2_status(v1=bad), True)
    drawer = resolve_v2_task("drawer_bowl")
    summary, _, _ = run(tmp_path, key="drawer_bowl", sim=FakeV2Sim(prompt=drawer.prompt, task_id=3))
    assert summary["status"] == "infrastructure_error"     # chained config, but the record is missing
    summary, _, _ = run(tmp_path, key="drawer_bowl", name="ok")
    assert summary["status"] == "diagnostic_complete"
    assert summary["collision_v1_record_only"]["mode"] == "task_manipulation_v1"


def test_collision_record_v2_invariants():
    good = v2_status()
    assert runner._collision_record_v2(good)["mode"] == "generic_contact_v2"
    for mutate in (lambda s: s.update(collision={**s["collision"], "mode": "task_manipulation_v1"}),
                   lambda s: s.update(collision={**s["collision"], "enabled": False}),
                   lambda s: s.update(collision={**s["collision"], "stop_on_collision": True}),
                   lambda s: s.update(collision={**s["collision"], "collision_policy": "stop"}),
                   lambda s: s.update(libero_success=None),
                   lambda s: s.update(success=True),
                   lambda s: s.update(collision=generic_record(True, {"robot_scene": {}}), terminated=True)):
        status = v2_status()
        mutate(status)
        with pytest.raises(ValueError):
            runner._collision_record_v2(status)


def test_configuration_validation():
    for key in ("libero_10_t2", "bowl_plate", "drawer_bowl"):
        runner.validate_configuration(CONFIG, sim_config(key), resolve_v2_task(key))
    assert sim_config("libero_10_t2").collision_check_mode == "none"
    assert sim_config("drawer_bowl").collision_check_mode == "task_manipulation_v1"
    from dataclasses import replace
    with pytest.raises(ValueError):
        runner.validate_configuration(CONFIG, replace(sim_config(), stop_on_collision=True), T2)
    with pytest.raises(ValueError):
        runner.validate_configuration(CONFIG, replace(sim_config(), stop_on_success=False), T2)
    with pytest.raises(ValueError):
        runner.validate_configuration(CONFIG, replace(sim_config(), include_arm_joints=True), T2)
    with pytest.raises(ValueError):
        runner.validate_configuration(CONFIG, sim_config("libero_10_t2"), resolve_v2_task("drawer_bowl"))


def test_fresh_output_and_mode_refusals(tmp_path):
    (tmp_path / "out").mkdir()
    with pytest.raises(FileExistsError):
        run(tmp_path)
    # RUN-EXT (BI, the authorised edit [C-43]; user decision 2026-10-19 live-first): the relation smoke is no longer
    # refused (tests/test_agent_speedup_v2_chain.py runs it), and the live client is handed out only to an
    # explicit live run.
    with pytest.raises(ValueError, match="explicit live run"):
        runner._default_client_factory(False)
    assert runner._default_client_factory(False, live=True).__name__ == "AgentSpeedUpV2Client"


@pytest.mark.parametrize("key", ["libero_10_t2", "bowl_plate", "drawer_bowl"])
def test_preflight(key, tmp_path):
    result = runner.preflight(CONFIG, sim_config(key), task_key=key)
    task = resolve_v2_task(key)
    assert result["status"] == "preflight_only" and result["client_task_prompt"] == task.prompt
    assert result["lint_check"]["passed"] is True
    assert result["frozen_reset_sha256"] == task.frozen_reset_sha256[0]
    assert result["registry_prompt_is_v1_bowl_default"] is False
    assert result["api_calls"] == 0 and result["simulator_started"] is False
    assert not any(tmp_path.iterdir())


def test_cli_preflight_and_live_refusal():
    out = subprocess.run([str(ROOT / ".venv/bin/python"), str(ROOT / "scripts/evaluate_agent_speedup_v2.py"),
                          "--preflight", "--task", "libero_10_t2"], capture_output=True, text=True, cwd=ROOT)
    assert out.returncode == 0, out.stderr[-2000:]
    assert json.loads(out.stdout)["task"] == T2.prompt
    out = subprocess.run([str(ROOT / ".venv/bin/python"), str(ROOT / "scripts/evaluate_agent_speedup_v2.py"),
                          "--task", "libero_10_t2", "--output", "/nonexistent/x"], capture_output=True, text=True,
                         cwd=ROOT)
    assert out.returncode == 2 and "choose a mode" in out.stderr     # RUN-EXT (BI): live needs --live


def test_plumbing_policy_contract():
    client = runner.OfflinePlumbingClient(CONFIG, task_prompt=T2.prompt)
    policy = runner.PlumbingHoldPolicy(client)
    fake = FakeV2Sim(prompt=T2.prompt)
    obs = fake.reset(2, 0)
    decision = policy.act(obs)
    with pytest.raises(RuntimeError):
        policy.act(obs)
    after, _ = fake.step(decision.action)
    policy.observe_transition(obs, decision, after)
    with pytest.raises(RuntimeError):
        policy.observe_transition(obs, decision, after)
    total = sum(steps for _, steps, _ in runner.PLUMBING_SCRIPT)
    actions = [runner.PlumbingHoldPolicy(client)._phase(i)[3] for i in range(total + 3)]
    assert all(len(a) == 7 and all(-1 <= v <= 1 for v in a) for a in actions)
    assert actions[-1] == (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0)
