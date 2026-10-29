"""Fair paired orchestration and artifact checks, without model/simulator calls."""

import importlib.util
import json
from pathlib import Path
import sys

import pytest

from gpt_field.compact_subscription import compact_operational_guidance


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("matched_compact_comparison_test", SCRIPTS / "compare_optimized_compact.py")
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def arm(directory, mode, *, success=True, source=None, runner_hash="runner-fixture"):
    source = source or {"policy.py": "same-source"}
    policy = {"controller_mode": mode, "execution_mode": "tracking" if mode == "hybrid" else "fixed",
        "feedback_history": "block_summary", "direct_action_steps": 5, "field_update_steps": 30,
        "field_representation": "analytic-goal-rbf-v1", "centered_placement_guidance": True,
        "grasp_settle_guidance": True, "grid": {"size": 33, "spacing_m": .005, "frame": "world"},
        "planner": {"horizon": 10, "step_m": .01}, "field_execution_steps": 10,
        "adapter_max_rotation_rad": .15, "recent_image_observations": 2,
        "initial_history_settings": {"image_history_limit": 2, "messages": 0},
        "reasoning_effort": "medium", "success_feedback_to_policy": False,
        "reward_feedback_to_policy": False, "object_state_feedback_to_policy": False}
    manifest = {"configuration": {"simulator": {"seed": 7, "suite": "libero_spatial", "stop_on_success": True},
        "stopping_rule": driver.runner._stopping_rule(False), "stop_on_model_complete": False,
        "software": {"source_sha256": source}, "runner": {"sha256": runner_hash}, "policy": policy,
        "api": {"model": "gpt-6-astra", "timeout_s": 180},
        "task": "bowl task", "task_id": 0, "episode_index": 0}}
    write(directory / "manifest.json", manifest)
    summary = {"status": "success" if success else "subscription_error", "error": None if success else {"code": 503},
        "benchmark_completed": success, "libero_success_observed": success, "libero_success": True if success else None,
        "evaluation_completed": success, "model_declared_complete": False,
        "control_elapsed_s": 10. if mode == "hybrid" else 20., "elapsed_s": 25.,
        "startup_to_initial_observation_s": 2., "timing_components": {}, "steps": 1,
        "coarse_field_actions": int(mode == "hybrid"), "contact_direct_actions": int(mode == "direct"),
        "api_calls": 1, "local_plan_count": int(mode == "hybrid"), "billing": "chatgpt_subscription",
        "api_key_calls": 0, "budget": {"input_tokens": 100, "output_tokens": 20}}
    write(directory / "summary.json", summary)
    write(directory / "simulator.json", {"metadata": {"control_freq_hz": 20, "image_size": 224,
        "controller_sources": {"osc": "same-hash"}, "packages": {"mujoco": "fixture"}}})
    reset = {"event": "reset", "observation": {"image": "same-camera", "state": [0., 0., 1.]}}
    executed = {"event": "executed", "step": 0, "success": success}
    (directory / "trace.jsonl").write_text(json.dumps(reset) + "\n" + json.dumps(executed) + "\n")
    instructions = f"{mode} controller-specific output format. " + driver._matched_guidance()
    request = {"billing": "chatgpt_subscription", "payload": {"model": "gpt-6-astra",
        "reasoning": {"effort": "medium"}, "instructions": instructions}}
    response = {"status": "completed", "billing": "chatgpt_subscription", "model": "gpt-6-astra",
        "reasoning": {"effort": "medium"}, "usage": {"input_tokens": 100}}
    write(directory / "requests/001.request.json", request)
    write(directory / "requests/001.response.json", response)
    return summary


def pair(directory):
    for mode in ("hybrid", "direct"):
        arm(directory / mode, mode)
    manifest = {"matched_libero_guidance": True, "order": ["hybrid", "direct"],
        "operational_guidance_sha256": driver.hashlib.sha256(driver._matched_guidance().encode()).hexdigest(),
        "source_sha256": {"policy.py": "same-source"}, "runner_sha256": "runner-fixture",
        "arm_directories": {mode: str((directory / mode).resolve()) for mode in ("hybrid", "direct")}}
    write(directory / "comparison_manifest.json", manifest)
    refresh_pair_hashes(directory)


def refresh_pair_hashes(directory):
    path = directory / "comparison_manifest.json"
    value = json.loads(path.read_text())
    value["arm_manifest_sha256"] = {mode: driver._file_sha256(directory / mode / "manifest.json")
                                    for mode in ("hybrid", "direct")}
    write(path, value)


def test_actual_success_and_identical_guidance_allow_ratio_despite_method_specific_text(tmp_path):
    pair(tmp_path)
    result = driver.compare(tmp_path, matched_libero_guidance=True)
    assert result["valid_comparison"] and all(result["checks"].values())
    assert result["hybrid_speedup"] == 2.
    assert result["hybrid_seconds_saved"] == 10.
    assert result["hybrid_time_reduction_percent"] == 50.
    for metrics in result["arms"].values():
        assert metrics["operational_guidance_occurrences_per_request"] == [1]
        assert metrics["actual_libero_success"]


@pytest.mark.parametrize("mutation,failed_check", [
    ("guidance_flag", "both_guidance_flags_enabled"),
    ("guidance_text", "identical_operational_guidance_in_every_request"),
    ("guidance_twice", "identical_operational_guidance_in_every_request"),
    ("model_stop", "libero_only_completion"),
    ("model_claim", "both_actual_libero_success"),
    ("trace_failure", "both_actual_libero_success"),
    ("api_config", "same_api_configuration"),
    ("images", "two_image_pairs_and_no_hidden_feedback"),
    ("hidden_feedback", "two_image_pairs_and_no_hidden_feedback"),
    ("physical_simulator", "same_simulator_physical_metadata"),
    ("sources", "same_sources"),
    ("history", "same_summary_history"),
])
def test_unfair_or_unverified_arms_cannot_claim_completed_speedup(tmp_path, mutation, failed_check):
    pair(tmp_path)
    directory = tmp_path / "direct"
    path = directory / "manifest.json"
    manifest = json.loads(path.read_text())
    config = manifest["configuration"]
    if mutation == "guidance_flag": config["policy"]["grasp_settle_guidance"] = False
    elif mutation == "model_stop": config["stop_on_model_complete"] = True
    elif mutation == "api_config": config["api"]["timeout_s"] = 90
    elif mutation == "images": config["policy"]["recent_image_observations"] = 1
    elif mutation == "hidden_feedback": config["policy"]["success_feedback_to_policy"] = True
    elif mutation == "sources": config["software"]["source_sha256"]["policy.py"] = "different"
    elif mutation == "history": config["policy"]["feedback_history"] = "full"
    elif mutation.startswith("guidance_"):
        p = directory / "requests/001.request.json"; request = json.loads(p.read_text())
        request["payload"]["instructions"] = ("Different grasp instructions" if mutation == "guidance_text" else
            request["payload"]["instructions"] + driver._matched_guidance())
        write(p, request)
    elif mutation == "model_claim":
        p = directory / "summary.json"; summary = json.loads(p.read_text())
        summary.update(status="model_declared_complete", model_declared_complete=True,
                       libero_success=None, libero_success_observed=False, evaluation_completed=False)
        write(p, summary)
    elif mutation == "trace_failure":
        p = directory / "trace.jsonl"; rows = [json.loads(line) for line in p.read_text().splitlines()]
        rows[-1]["success"] = False; p.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    elif mutation == "physical_simulator":
        p = directory / "simulator.json"; value = json.loads(p.read_text())
        value["metadata"]["control_freq_hz"] = 10; write(p, value)
    write(path, manifest)
    refresh_pair_hashes(tmp_path)
    result = driver.compare(tmp_path, matched_libero_guidance=True)
    assert result["checks"][failed_check] is False
    assert not result["valid_comparison"]
    assert result["hybrid_speedup"] is result["hybrid_seconds_saved"] is result["hybrid_time_reduction_percent"] is None


def test_matched_report_rejects_substituted_historical_success(tmp_path):
    pair(tmp_path)
    with pytest.raises(ValueError, match="fresh paired"):
        driver.compare(tmp_path, matched_libero_guidance=True, direct_directory=tmp_path / "direct")
    manifest = json.loads((tmp_path / "comparison_manifest.json").read_text())
    manifest["arm_manifest_sha256"]["direct"] = "historical-manifest"
    write(tmp_path / "comparison_manifest.json", manifest)
    result = driver.compare(tmp_path, matched_libero_guidance=True)
    assert not result["checks"]["fresh_pair_provenance"]


def test_legacy_model_declared_completion_reporting_remains_supported(tmp_path):
    pair(tmp_path)
    p = tmp_path / "direct/summary.json"; summary = json.loads(p.read_text())
    summary.update(status="model_declared_complete", model_declared_complete=True,
                   libero_success=None, libero_success_observed=False, evaluation_completed=False)
    write(p, summary)
    result = driver.compare(tmp_path, direct_directory=tmp_path / "direct")
    assert result["valid_comparison"]
    assert result["hybrid_speedup"] == 2.


def test_real_preflights_share_guidance_without_credentials_or_simulation():
    sim = driver.LiberoConfig(suite="libero_spatial", seed=7, max_steps=None)
    payloads = []
    for mode in ("hybrid", "direct"):
        result = driver.runner.preflight(driver.runner.configuration(), sim, field_representation="compact",
            reasoning_effort="medium", stop_on_model_complete=False, controller_mode=mode,
            execution_mode="tracking" if mode == "hybrid" else "fixed", feedback_history="block_summary",
            centered_placement_guidance=True, grasp_settle_guidance=True)
        assert result["api_calls"] == result["api_key_calls"] == 0
        assert not result["simulator_started"] and not result["credentials_checked"]
        assert result["payload"]["instructions"].count(driver._matched_guidance()) == 1
        payloads.append(result["payload"])
    assert payloads[0]["instructions"] != payloads[1]["instructions"]
    assert payloads[0]["text"] != payloads[1]["text"]


def test_incomplete_second_arm_before_control_start_retains_failure_without_ratio(tmp_path):
    pair(tmp_path)
    p = tmp_path / "direct/summary.json"
    summary = json.loads(p.read_text())
    summary.update(status="infrastructure_error", benchmark_completed=False,
                   evaluation_completed=False, libero_success=None, libero_success_observed=False,
                   control_elapsed_s=None, startup_to_initial_observation_s=None, steps=0,
                   error={"code": "simulator_startup"})
    write(p, summary)
    (tmp_path / "direct/trace.jsonl").write_text("")
    result = driver.compare(tmp_path, matched_libero_guidance=True)
    assert not result["valid_comparison"]
    assert result["hybrid_speedup"] is None
    assert result["arms"]["direct"]["error"] == {"code": "simulator_startup"}
    assert result["censored_direct_observation"]["observed_time_beyond_hybrid_completion_s"] is None


@pytest.mark.parametrize("matched", [False, True])
def test_driver_launches_both_fresh_arms_with_matched_settings(tmp_path, monkeypatch, matched):
    output = tmp_path / "fresh"
    calls, preflights = [], []
    monkeypatch.setattr(driver.runner, "preflight", lambda config, sim, **kw: preflights.append(kw))
    monkeypatch.setattr(driver, "source_manifest", lambda: {"source_sha256": {"policy.py": "same-source"}})
    from gpt_field import chatgpt_auth
    monkeypatch.setattr(chatgpt_auth, "SubscriptionCredentials", lambda **kw: type("Credentials", (), {"access_token": lambda self: "unused"})())

    def execute(path, **kwargs):
        assert not path.exists()
        calls.append((path, kwargs))
        return arm(path, kwargs["controller_mode"], runner_hash=driver._file_sha256(driver.runner.__file__))

    monkeypatch.setattr(driver.runner, "execute_episode", execute)
    assert driver.main(["--output", str(output), *( ["--matched-libero-guidance"] if matched else [])]) == 0
    assert [p.name for p, _ in calls] == ["hybrid", "direct"]
    assert [kw["execution_mode"] for _, kw in calls] == ["tracking", "fixed"]
    for settings in preflights + [kw for _, kw in calls]:
        assert settings["reasoning_effort"] == "medium"
        assert settings["feedback_history"] == "block_summary"
        assert settings["stop_on_model_complete"] is (not matched)
        assert settings.get("centered_placement_guidance", False) is matched
        assert settings.get("grasp_settle_guidance", False) is matched
    assert calls[0][1]["sim_config"] == calls[1][1]["sim_config"]
    if matched:
        manifest = json.loads((output / "comparison_manifest.json").read_text())
        assert manifest["operational_guidance"] == compact_operational_guidance(
            centered_placement_guidance=True, grasp_settle_guidance=True)
        assert manifest["stop_on_model_complete"] is False


def test_failed_first_arm_is_retained_and_does_not_launch_service_retry(tmp_path, monkeypatch):
    output = tmp_path / "failed"
    calls = []
    monkeypatch.setattr(driver.runner, "preflight", lambda *args, **kw: None)
    from gpt_field import chatgpt_auth
    monkeypatch.setattr(chatgpt_auth, "SubscriptionCredentials", lambda **kw: type("Credentials", (), {"access_token": lambda self: "unused"})())

    def execute(path, **kwargs):
        calls.append(kwargs["controller_mode"])
        return arm(path, kwargs["controller_mode"], success=False)

    monkeypatch.setattr(driver.runner, "execute_episode", execute)
    assert driver.main(["--output", str(output), "--matched-libero-guidance"]) == 3
    assert calls == ["hybrid"]
    result = json.loads((output / "comparison.json").read_text())
    assert result["arms"]["hybrid"]["status"] == "subscription_error"
    assert result["arms"]["hybrid"]["steps"] == 1
    assert result["arms"]["direct"] is None
    assert result["hybrid_speedup"] is None
