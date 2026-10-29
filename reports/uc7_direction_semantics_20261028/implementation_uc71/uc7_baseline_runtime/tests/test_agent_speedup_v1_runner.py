"""agent_SpeedUp_v1 runner and recorder plumbing with fake clients, policies and simulators only."""

from dataclasses import replace
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import numpy as np
import pytest

from agent_speedup_fakes import DRAWER_PROMPT, FakeLiberoSim, png_url
from gpt_field.astra import APIError
from gpt_field.policy import Decision


ROOT = Path(__file__).resolve().parents[1]
FINAL_ARTIFACTS = ("summary.json", "runner_summary.json", "ledger.json", "agent_stats.json", "task_evaluation.json",
                   "episodes.jsonl", "run_state.json")
spec = importlib.util.spec_from_file_location("evaluate_agent_speedup_v1", ROOT / "scripts/evaluate_agent_speedup_v1.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
CONFIG = runner.configuration()
LEAK = "SECRET.TOKEN-abc123_must_not_leak"


def rows(path):
    def reject(value):
        raise ValueError(f"non-finite JSON constant {value}")
    return [json.loads(line, parse_constant=reject) for line in path.read_text().splitlines()]


def explain(output, result):
    trace = output / "error_traceback.txt"
    return f"{result['error']}\n{trace.read_text() if trace.is_file() else ''}"


class FakeClient:
    instances = []
    prompt_version = "fake-agent-prompt"

    def __init__(self, config=None, *, token_provider, task_key="bowl_plate", target_description=None,
                 transport=None, log_dir=None, clock=time.monotonic):
        self.config, self.token_provider, self.task_key = config, token_provider, task_key
        self.task_prompt = runner.resolve_distance_task(task_key).prompt
        self.target_description, self.transport, self.log_dir = target_description, transport, log_dir
        self.reasoning_effort = "medium"
        self.calls, self.payloads, self.tokens = [], [], []
        self._used_access_tokens = set()
        self.instances.append(self)

    def _kinds(self):
        return {kind: self.calls.count(kind) for kind in ("decision", "pointing_coarse", "pointing_fine")}

    def budget_snapshot(self):
        return {"billing": "chatgpt_subscription", "api_calls": len(self.calls), "api_key_calls": 0}

    def history_stats(self):
        return {"protocol": "fake-stateless", "accepted_requests": len(self.calls), "messages": 0,
                "retained_image_pairs": 0, "calls_by_kind": self._kinds()}

    def build_payload(self, request, grid=None, mode="decision"):
        self.payloads.append((mode, request))
        images = [{"type": "input_image", "image_url": value, "detail": "high"}
                  for value in request.values() if isinstance(value, str) and value.startswith("data:image/")]
        return {"model": "gpt-6-astra", "service_tier": "default", "store": False, "instructions": f"fake {mode}",
                "input": [{"role": "user", "content": [{"type": "input_text", "text": mode}, *images]}],
                "text": {"format": {"type": "json_schema", "name": mode, "strict": True, "schema": {}}},
                "stream": True, "reasoning": {"effort": "medium"}, "include": ["reasoning.encrypted_content"]}

    def _call(self, kind):
        token = self.token_provider()
        self.tokens.append(token)
        self._used_access_tokens.add(token)
        self.calls.append(kind)

    def decide(self, request):
        self._call("decision")
        return {"situation": "bowl left of the fingers", "skill": "reach",
                "params": {"target": None, "strategy": None, "side": None, "approach": None,
                           "waypoint_offset_m": [0.0, 0.0, -0.02], "more_waypoints_m": None, "max_travel_m": None}}

    def point(self, request):
        self._call("pointing_" + request["stage"])
        return {"cell_id": 36} if request["stage"] == "coarse" else {"point_id": 28}


class FakeCalibration:
    def to_dict(self):
        return {"open_gap_mm": 77.4, "source": "online"}


class FakePolicy:
    instances = []

    def __init__(self, client, *, local_retries=2, fail_at=None, error=None, nan_at=None, float32=False):
        self.client, self.local_retries, self.grid = client, local_retries, None
        self.fail_at, self.error, self.nan_at, self.float32 = fail_at, error, nan_at, float32
        self.events, self.index, self._pending, self._ledger = [], 0, None, []
        self.instances.append(self)

    def reset(self):
        self.events.append("reset")
        self.index, self._ledger = 0, []

    def act(self, observation):
        self.events.append("act")
        if self.fail_at == self.index:
            raise self.error
        refreshed = self.index % 4 == 0
        calls, routing, pointing, outcome, latency = [], None, [], None, None
        if refreshed:
            choice = self.client.decide({"state_summary": {"eef_xyz_m": list(observation.state[:3])},
                                         "external_image": observation.image, "wrist_image": observation.wrist_image})
            calls.append("decision")
            routing = {"skill": choice["skill"], "params": choice["params"], "situation": choice["situation"]}
            latency = 0.25
            if self.index:
                outcome = {"skill": "reach", "status": "succeeded", "reason": "goal_reached",
                           "evidence": {"steps": 4, "mark_uv": [300.0, 200.0]}}
                self._ledger.append({"skill": "reach", "status": "succeeded"})
        elif self.index % 4 == 1:
            coarse = self.client.point({"stage": "coarse", "image": observation.wrist_image})
            fine = self.client.point({"stage": "fine", "image": observation.wrist_image})
            calls += ["pointing_coarse", "pointing_fine"]
            pointing = [{"stage": "coarse", **coarse}, {"stage": "fine", **fine, "uv": [262.0, 310.0]}]
            latency = 0.5
        values = (0.0, 0.0, -0.2, 4e-9, 0.0, 0.0, -1.0)
        action = tuple(np.float32(v) for v in values) if self.float32 else values
        diagnostics = {"step": self.index, "controller": "agent_skill", "refreshed": refreshed, "routing": routing,
                       "api_latency_s": latency, "api_calls_in_act": calls, "api_request_key": None,
                       "api_request_log": None, "execution_kind": "agent_skill:reach", "macro": "reach",
                       "primitive": "waypoint_reach", "macro_step": self.index % 4, "update_steps": 1,
                       "history_stats": self.client.history_stats(), "feedback": None, "plan_refreshed": False,
                       "outcome": outcome, "pointing": pointing}
        if self.nan_at == self.index:
            diagnostics["bad"] = float("nan")
        self._pending = Decision(action, diagnostics)
        return self._pending

    def observe_transition(self, before, decision, after):
        assert decision is self._pending
        self.events.append("observe")
        self.index += 1
        delta = np.subtract(after.state[:3], before.state[:3])
        return {"step": self.index - 1, "measured_delta_m": [float(v) for v in delta], "primitive": "waypoint_reach"}

    def agent_stats(self):
        return {"acts": self.events.count("act"), "ledger_entries": len(self._ledger),
                "local_retries": self.local_retries}

    @property
    def ledger(self):
        return list(self._ledger)

    @property
    def calibration(self):
        return FakeCalibration() if self.index else None


class FakeRecorder:
    instances = []

    def __init__(self, path, **kwargs):
        self.path, self.kwargs, self.events = path, kwargs, []
        self.instances.append(self)

    def observe(self, obs, **kwargs):
        self.events.append(("observe", kwargs))

    def decision(self, obs, decision, **kwargs):
        self.events.append(("decision", kwargs))

    def executed(self, obs, **kwargs):
        self.events.append(("executed", kwargs))

    def close(self, **kwargs):
        self.events.append(("close", kwargs))


class OfflineTransport:
    offline_mock = True

    def __call__(self, *args):
        raise AssertionError("fake client never sends")


def sim_factory(**kwargs):
    seen = []

    def build(config):
        seen.append(os.environ.get("OPENAI_API_KEY"))
        return FakeLiberoSim(config, **kwargs)
    build.seen = seen
    return build


def run(tmp_path, *, name="trial", smoke=False, **kwargs):
    task_key = kwargs.get("task_key", "bowl_plate")
    options = {"config": CONFIG, "sim_config": runner.build_sim_config(task_key), "task_key": task_key,
               "episode_index": 0, "client_factory": FakeClient,
               "policy_factory": lambda client, **kw: FakePolicy(client, **kw),
               "sim_factory": sim_factory(success_at=5), "recorder_factory": FakeRecorder}
    if not smoke:
        options["token_provider"] = lambda: LEAK
    options.update(kwargs)
    output = tmp_path / name
    return output, runner.execute(output, smoke=smoke, **options)


def test_success_path_writes_complete_artifacts(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-must-not-reach-simulator")
    sim = sim_factory(success_at=5)
    output, result = run(tmp_path, sim_factory=sim)
    assert sim.seen == [None] and os.environ["OPENAI_API_KEY"] == "sk-must-not-reach-simulator"
    assert result["status"] == "success" and result["steps"] == 5
    assert result["evaluation_completed"] and result["task_success"] and result["success_rate"] == 1.0
    assert result["libero_success"] is True and result["completion_time_eligible"]
    assert result["skill_decision_acts"] == 2 and result["pointing_results"] == 2
    assert result["macro_outcome_counts"] == {"reach:succeeded": 1}
    assert result["calls_by_kind"] == {"decision": 2, "pointing_coarse": 1, "pointing_fine": 1}
    assert result["api_calls"] == 4 and result["billing"] == "chatgpt_subscription" and result["evaluation_scored"]
    assert result["calibration"] == {"open_gap_mm": 77.4, "source": "online"}
    assert result["measured_feedback_count"] == 5 and result["gripper_open_actions"] == 5
    timings = result["timing_components"]
    assert set(timings) >= {"decision_s", "model_pipeline_s", "simulator_step_s", "video_render_s"}
    assert timings["model_pipeline_s"] == pytest.approx(1.0)
    assert result["control_without_video_s"] == pytest.approx(result["control_elapsed_s"] - timings["video_render_s"])
    evaluation = json.loads((output / "task_evaluation.json").read_text())
    assert set(evaluation) == {"status", "task_success", "libero_success", "collision", "failure_reason",
                               "evaluation_completed", "success_rate", "scored_episodes", "incomplete_episodes",
                               "completion_time_eligible"}
    assert json.loads((output / "collision.json").read_text())["collision_policy"] == "record_only"
    assert json.loads((output / "agent_stats.json").read_text())["acts"] == 5
    assert json.loads((output / "ledger.json").read_text()) == [{"skill": "reach", "status": "succeeded"}]
    assert json.loads((output / "run_state.json").read_text())["stage"] == "finished"
    assert json.loads((output / "summary.json").read_text()) == json.loads((output / "runner_summary.json").read_text())
    assert rows(output / "episodes.jsonl")[0]["status"] == "success"
    check = json.loads((output / "initial_observation_check.json").read_text())
    assert check["matched"] is None and len(check["sha256"]) == 64
    assert not (output / "error_traceback.txt").exists() and result["error_traceback"] is None
    policy, recorder = FakePolicy.instances[-1], FakeRecorder.instances[-1]
    assert policy.events[0] == "reset" and policy.local_retries == 2
    assert [event for event, _ in recorder.events] == ["observe"] + ["decision", "executed"] * 5 + ["close"]
    assert "AGENT SPEEDUP V1" in recorder.kwargs["episode_id"] and recorder.kwargs["offline_smoke"] is False
    assert recorder.events[2][1]["history_stats"]["protocol"] == "fake-stateless"
    assert FakeClient.instances[-1].tokens == [LEAK] * 4
    assert LEAK not in (output / "trace.jsonl").read_text()


def test_trace_rows_are_json_safe_and_cover_agent_events(tmp_path):
    output, result = run(tmp_path, policy_factory=lambda client, **kw: FakePolicy(client, float32=True, **kw))
    assert result["status"] == "success"
    trace = rows(output / "trace.jsonl")
    events = [row["event"] for row in trace]
    assert events[0] == "reset"
    assert {"skill_decision", "pointing", "macro_outcome", "action", "executed", "feedback"} <= set(events)
    actions = [row for row in trace if row["event"] == "action"]
    assert len(actions) == 5 and all(type(value) is float for row in actions for value in row["action"])
    assert all(row["diagnostics"]["controller"] == "agent_skill" for row in actions)
    decision = next(row for row in trace if row["event"] == "skill_decision")
    assert decision["routing"]["skill"] == "reach" and decision["api_calls_in_act"] == ["decision"]
    assert decision["source"] == "model_prediction"
    pointing = [row["result"] for row in trace if row["event"] == "pointing"]
    assert pointing == [{"stage": "coarse", "cell_id": 36}, {"stage": "fine", "point_id": 28, "uv": [262.0, 310.0]}]
    outcome = next(row for row in trace if row["event"] == "macro_outcome")
    assert outcome["step"] == 4 and outcome["outcome"]["reason"] == "goal_reached"
    order = [row["event"] for row in trace if row.get("step") == 4]
    assert order.index("macro_outcome") < order.index("skill_decision") < order.index("action")
    executed = [row for row in trace if row["event"] == "executed"]
    assert executed[-1]["success"] is True and executed[0]["macro"] == "reach"


def test_operator_stop_ends_at_feedback_boundary(tmp_path):
    output = tmp_path / "trial"

    class StoppingSim(FakeLiberoSim):
        def step(self, action):
            result = super().step(action)
            if len(self.actions) == 3:
                (output / "operator_stop.request.json").write_text("{}")
            return result

    _, result = run(tmp_path, sim_factory=lambda config: StoppingSim(config))
    assert result["status"] == "user_stop" and result["steps"] == 3
    assert result["error"]["type"] == "OperatorStopRequested" and result["pending_action_index"] is None
    assert result["error"]["source"] == "file"
    assert not result["evaluation_completed"] and result["incomplete_episodes"] == 1
    assert result["measured_feedback_count"] == 3
    stop = [row for row in rows(output / "trace.jsonl") if row["event"] == "operator_stop"]
    assert stop == [{"event": "operator_stop", "source": "file", "step": 3, "feedback_count": 3,
                     "pending_action_index": None, "request_file": "operator_stop.request.json",
                     "stage": "observed"}]
    for name in FINAL_ARTIFACTS:
        assert (output / name).is_file(), name
    assert rows(output / "episodes.jsonl")[0]["status"] == "user_stop"
    assert json.loads((output / "task_evaluation.json").read_text())["status"] == "user_stop"
    assert result["operator_stop_command"] == f"touch {output / 'operator_stop.request.json'}"
    assert FakeLiberoSim.instances[-1].closed


def test_initial_observation_mismatch_stops_before_policy_reset(tmp_path):
    output, result = run(tmp_path, expected_initial_observation_sha256="0" * 64)
    assert result["status"] == "infrastructure_error" and result["steps"] == 0
    assert result["error"]["type"] == "ValueError" and result["error"]["stage"] == "starting"
    policy, client = FakePolicy.instances[-1], FakeClient.instances[-1]
    assert policy.events == [] and client.calls == []
    check = json.loads((output / "initial_observation_check.json").read_text())
    assert check["matched"] is False and check["expected_sha256"] == "0" * 64
    assert "no inference performed" in (output / "error_traceback.txt").read_text()
    assert FakeLiberoSim.instances[-1].closed and FakeLiberoSim.instances[-1].actions == []


def test_initial_observation_match_is_recorded(tmp_path):
    first, _ = run(tmp_path, name="first")
    sha = json.loads((first / "initial_observation_check.json").read_text())["sha256"]
    second, result = run(tmp_path, name="second", expected_initial_observation_sha256=sha)
    assert result["status"] == "success"
    assert json.loads((second / "initial_observation_check.json").read_text())["matched"] is True


def test_errors_write_redacted_traceback(tmp_path):
    error = RuntimeError(f"transport failed: Authorization: Bearer {LEAK} raw {LEAK} eyJhbGciOi.eyJzdWIiOiJ4.c2lnbmF0dXJl")
    policy = lambda client, **kw: FakePolicy(client, fail_at=2, error=error, **kw)
    output, result = run(tmp_path, policy_factory=policy)
    assert result["status"] == "infrastructure_error" and result["steps"] == 2
    assert result["error"] == {"type": "RuntimeError", "stage": "deciding",
                               "message": "Diagnostic stopped; no retry or API-key fallback"}
    assert result["error_traceback"] == "error_traceback.txt"
    text = (output / "error_traceback.txt").read_text()
    assert "RuntimeError" in text and "transport failed" in text and "Bearer [REDACTED]" in text
    assert LEAK not in text and "eyJhbGciOi" not in text
    assert LEAK not in (output / "summary.json").read_text()
    assert json.loads((output / "run_state.json").read_text())["status"] == "infrastructure_error"
    assert FakeLiberoSim.instances[-1].closed


def test_subscription_errors_and_non_finite_diagnostics(tmp_path):
    policy = lambda client, **kw: FakePolicy(client, fail_at=1, error=APIError("quota"), **kw)
    _, result = run(tmp_path, name="api", policy_factory=policy)
    assert result["status"] == "subscription_error" and result["error"]["type"] == "APIError"
    output, result = run(tmp_path, name="nan", policy_factory=lambda client, **kw: FakePolicy(client, nan_at=1, **kw))
    assert result["status"] == "infrastructure_error" and result["steps"] == 1
    assert "Out of range float values are not JSON compliant" in (output / "error_traceback.txt").read_text()
    rows(output / "trace.jsonl")


def test_non_agent_controller_and_unsafe_stats_are_infrastructure_errors(tmp_path):
    class LegacyPolicy(FakePolicy):
        def act(self, observation):
            decision = super().act(observation)
            return Decision(decision.action, {**decision.diagnostics, "controller": "contact_direct"})

    _, result = run(tmp_path, name="legacy", policy_factory=lambda client, **kw: LegacyPolicy(client, **kw))
    assert result["status"] == "infrastructure_error" and result["steps"] == 0

    class UnsafeStats(FakePolicy):
        def agent_stats(self):
            return {"value": np.float32(1.0)}

    # A failed metadata read is recorded separately; it never overrides the episode outcome (success here).
    output, result = run(tmp_path, name="stats", policy_factory=lambda client, **kw: UnsafeStats(client, **kw))
    assert result["status"] == "success" and result["steps"] == 5
    assert any(item["stage"] == "agent_stats" for item in result["cleanup_errors"])
    assert json.loads((output / "agent_stats.json").read_text()) is None
    # ...but it still turns a non-outcome status into infrastructure_error.
    unsafe_api = lambda client, **kw: type("UnsafeAPI", (UnsafeStats,), {})(client, fail_at=2, error=APIError("quota"), **kw)
    _, result = run(tmp_path, name="stats_api", policy_factory=unsafe_api)
    assert result["status"] == "infrastructure_error" and result["error"]["type"] == "APIError"


def test_manifest_identity_and_frozen_source(tmp_path):
    output, _ = run(tmp_path, target_description="  the round bowl  ", local_retries=1, video=False)
    manifest = json.loads((output / "manifest.json").read_text())
    identity = manifest["configuration"]
    assert manifest["identity_hash"] == runner.digest(identity)
    assert identity["mode"] == runner.LIVE_MODE and identity["evaluation_scored"] is True
    assert identity["policy"]["class"].endswith("FakePolicy") and identity["policy"]["grid"] is None
    assert identity["policy"]["field_update_steps"] is None and identity["policy"]["controller"] == "agent_skill"
    assert identity["client"]["prompt_version"] == "fake-agent-prompt"
    assert identity["client"]["reasoning_effort"] == "medium"
    assert identity["client"]["history_protocol"] == "fake-stateless"
    assert identity["client"]["target_description"] == "the round bowl"
    agent = identity["agent_speedup_v1"]
    assert agent["rulings_assumed"] == {"R-1_online_image_jacobian": True, "R-2_wrist_mount_facts": True,
                                        "R-3_evaluator_tuned_constants": False,
                                        "R-4_episode_continuation_signal": False,
                                        "R-1x_external_image_axes_from_own_motion": True,
                                        "R-5_robot_base_position": True}
    assert agent["local_retries"] == 1 and agent["video"] is False and agent["constant_labels"]
    assert agent["versions"]["ORDERED_TRANSPORT_VERSION"] == "chatgpt-subscription-http-sse-ordered-v1"
    assert set(agent["versions"]) >= {"AGENT_SPEEDUP_V1_VERSION", "AGENT_PROMPT_VERSION", "MARK_PROTOCOL_VERSION",
                                      "CONTACT_SKILLS_VERSION", "WRIST_SERVO_VERSION"}
    assert identity["video"] is False and identity["local_retries"] == 1
    sim = identity["simulator"]
    assert (sim["render_size"], sim["image_size"], sim["seed"], sim["max_steps"]) == (512, 512, 7, None)
    assert sim["collision_check_mode"] == "task_manipulation_v1" and sim["stop_on_collision"] is False
    assert sim["python_executable"] == runner.DEFAULT_SIM_PYTHON and sim["include_arm_joints"] is False
    frozen = output / "frozen_source"
    for path in (runner.RUNNER_PATH, *runner.HELPER_SCRIPTS):
        assert (frozen / Path(path).name).read_bytes() == (ROOT / path).read_bytes()
    helpers = {item["path"]: item["sha256"] for item in identity["runner_helpers"]}
    assert helpers["scripts/evaluate_bowl_field_history.py"] == hashlib.sha256(
        (ROOT / "scripts/evaluate_bowl_field_history.py").read_bytes()).hexdigest()
    hashes = identity["software"]["source_sha256"]
    assert {"experiment.py", "ordered_transport.py", "agent_speedup_v1_video.py"} <= set(hashes)
    for name, sha in hashes.items():
        assert hashlib.sha256((frozen / name).read_bytes()).hexdigest() == sha
    assert FakeRecorder.instances[-1].path.name == "episode.mp4"


def test_no_video_uses_null_recorder(tmp_path):
    output, result = run(tmp_path, recorder_factory=None, video=False)
    assert result["status"] == "success" and result["video"] is None and result["video_enabled"] is False
    assert not (output / "episode.mp4").exists() and not (output / "episode.events.jsonl").exists()


def test_output_exists_is_refused(tmp_path, capsys):
    existing = tmp_path / "trial"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        run(tmp_path)
    assert list(existing.iterdir()) == []
    with pytest.raises(SystemExit) as exit_info:
        runner.main(["--output", str(existing)])
    assert exit_info.value.code == 2 and "already exists" in capsys.readouterr().err


def test_live_execution_requires_token_provider(tmp_path):
    with pytest.raises(runner.MissingCredentials):
        run(tmp_path, token_provider=None)
    assert not (tmp_path / "trial").exists()


def test_drawer_task_uses_goal_suite(tmp_path):
    sim = sim_factory(success_at=2, task_id=3, prompt=DRAWER_PROMPT)
    output, result = run(tmp_path, task_key="drawer_bowl", sim_factory=sim)
    assert result["status"] == "success" and result["suite"] == "libero_goal" and result["task_id"] == 3
    assert FakeClient.instances[-1].task_key == "drawer_bowl"
    assert json.loads((output / "manifest.json").read_text())["configuration"]["simulator"]["suite"] == "libero_goal"


def test_smoke_mode_is_offline_unscored_and_capped(tmp_path):
    output, result = run(tmp_path, smoke=True, smoke_steps=6, transport=OfflineTransport(),
                         sim_factory=sim_factory(success_at=None))
    assert result["status"] == "diagnostic_complete" and result["steps"] == 6 and result["diagnostic_complete"]
    assert result["billing"] == "offline_mock" and result["evaluation_scored"] is False
    assert not result["evaluation_completed"] and result["success_rate"] is None and result["scored_episodes"] == 0
    client = FakeClient.instances[-1]
    assert client.tokens and set(client.tokens) == {runner.OFFLINE_TOKEN}
    assert isinstance(client.transport, OfflineTransport)
    identity = json.loads((output / "manifest.json").read_text())["configuration"]
    assert identity["mode"] == runner.SMOKE_MODE and identity["max_policy_steps"] == 6
    assert identity["stopping_rule"][0] == "6 scripted smoke actions"
    assert FakeRecorder.instances[-1].kwargs["offline_smoke"] is True
    sources = {row.get("source") for row in rows(output / "trace.jsonl") if row["event"] == "skill_decision"}
    assert sources == {"scripted_offline_smoke"}


def test_smoke_mode_rejects_credentials_and_live_transports(tmp_path):
    with pytest.raises(ValueError, match="offline OAuth fixture"):
        run(tmp_path, smoke=True, token_provider=lambda: LEAK, transport=OfflineTransport())
    with pytest.raises(ValueError, match="offline scripted transport"):
        run(tmp_path, smoke=True, transport=lambda *args: {})
    with pytest.raises(ValueError, match="smoke_steps"):
        run(tmp_path, smoke=True, smoke_steps=0, transport=OfflineTransport())
    assert not (tmp_path / "trial").exists()


def test_configuration_is_validated(tmp_path):
    base = runner.build_sim_config("bowl_plate")
    bad = [{"sim_config": runner.build_sim_config("drawer_bowl")},
           {"sim_config": replace(base, image_size=224)}, {"sim_config": replace(base, stop_on_collision=True)},
           {"sim_config": replace(base, collision_check_mode="none")},
           {"sim_config": replace(base, include_arm_joints=True)}, {"sim_config": replace(base, seed=8)},
           {"local_retries": True}, {"local_retries": 9}, {"episode_index": 50},
           {"expected_initial_observation_sha256": "ABC"}, {"target_description": "   "}]
    for change in bad:
        with pytest.raises(ValueError):
            run(tmp_path, **change)
    assert not (tmp_path / "trial").exists()


def test_preflight_with_fake_client_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "SimProcess", lambda *a, **k: pytest.fail("preflight started the simulator"))
    output = tmp_path / "trial"
    result = runner.preflight(CONFIG, runner.build_sim_config("bowl_plate"), output, client_factory=FakeClient)
    assert not output.exists()
    assert result["status"] == "preflight_only" and result["credentials_checked"] is False
    assert result["api_calls"] == 0 and result["files_written"] is False and result["simulator_started"] is False
    client = FakeClient.instances[-1]
    assert client.calls == [] and client.log_dir is None
    with pytest.raises(runner.MissingCredentials):
        client.token_provider()
    with pytest.raises(APIError):
        client.transport("endpoint", {}, {}, 1)
    modes = [mode for mode, _ in client.payloads]
    assert modes == ["decision", "pointing_coarse", "pointing_fine"]
    summary = client.payloads[0][1]["state_summary"]
    axes = summary["tool_axes_world"]
    assert (axes.get("approach_axis") or axes.get("approach_R2")) == [0.0, 0.0, -1.0]
    assert summary["finger_gap_mm"] == 80.0
    assert result["state_summary_source"]
    image = result["payload"]["input"][0]["content"][1]["image_url"]
    assert image.startswith("<data:image/png;base64 ") and "sha256=" in image
    assert "data:image/png;base64,iVBOR" not in json.dumps(result)
    assert result["agent_speedup_v1"]["rulings_assumed"]["R-1_online_image_jacobian"] is True
    assert set(result["pointing_payloads"]) == {"coarse", "fine"}
    json.dumps(result, allow_nan=False)


def test_main_preflight_reads_no_credentials(tmp_path, monkeypatch, capsys):
    import gpt_field.chatgpt_auth as auth

    class NoCredentials:
        def __init__(self, *args, **kwargs):
            pytest.fail("preflight read credentials")

    monkeypatch.setattr(auth, "SubscriptionCredentials", NoCredentials)
    monkeypatch.setattr(runner, "SimProcess", lambda *a, **k: pytest.fail("preflight started the simulator"))
    monkeypatch.setattr(runner, "_default_client_factory", lambda smoke: FakeClient)
    output = tmp_path / "trial"
    assert runner.main(["--preflight", "--task", "drawer_bowl", "--output", str(output)]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["task_id"] == 3 and printed["simulator"]["suite"] == "libero_goal"
    assert not output.exists()


def test_main_wires_live_and_smoke_arguments(tmp_path, monkeypatch):
    import gpt_field.chatgpt_auth as auth
    captured = []

    class Credentials:
        def __init__(self, store_dir=None, profile=None):
            self.store_dir, self.profile = store_dir, profile

        def access_token(self):
            return LEAK

    def fake_execute(output, **kwargs):
        captured.append((output, kwargs))
        return {"evaluation_completed": kwargs["video"] and not kwargs["smoke"], "diagnostic_complete": False}

    monkeypatch.setattr(auth, "SubscriptionCredentials", Credentials)
    monkeypatch.setattr(runner, "execute", fake_execute)
    link = tmp_path / "python-link"
    link.symlink_to(shutil.which("python3") or "/usr/bin/python3")
    assert runner.main(["--output", str(tmp_path / "a"), "--sim-python", str(link), "--profile", "p",
                        "--local-retries", "1"]) == 0
    output, kwargs = captured[-1]
    assert kwargs["sim_config"].python_executable == str(link)
    assert kwargs["token_provider"].__self__.profile == "p" and kwargs["smoke"] is False
    assert kwargs["local_retries"] == 1 and kwargs["video"] is True
    assert runner.main(["--output", str(tmp_path / "b"), "--no-video"]) == 3
    assert runner.main(["--output", str(tmp_path / "c"), "--sim-smoke", "--smoke-steps", "40"]) == 3
    _, kwargs = captured[-1]
    assert kwargs["smoke"] is True and kwargs["smoke_steps"] == 40
    assert kwargs["token_provider"] is runner.offline_token_provider
    for argv in (["--output", str(tmp_path / "d"), "--smoke-steps", "5"],
                 ["--output", str(tmp_path / "e"), "--sim-smoke", "--profile", "p"],
                 ["--preflight", "--sim-smoke"], [],
                 ["--output", str(tmp_path / "f"), "--local-retries", "7"]):
        with pytest.raises(SystemExit):
            runner.main(argv)


def test_real_recorder_renders_frames(tmp_path):
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is required for annotated MP4 recording")
    output, result = run(tmp_path, recorder_factory=None, sim_factory=sim_factory(success_at=3))
    assert result["status"] == "success" and result["video"] == "episode.mp4"
    assert (output / "episode.mp4").stat().st_size > 1000
    events = rows(output / "episode.events.jsonl")
    kinds = [event["event"] for event in events]
    assert kinds == ["observation", "skill_decision", "executed", "agent_action", "executed",
                     "agent_action", "executed", "terminal"]
    decision = events[1]
    assert decision["routing"]["skill"] == "reach" and decision["policy"] == "agent_speedup_v1"
    assert events[3]["pointing"][1]["uv"] == [262.0, 310.0] and events[3]["frame_count"] == 4
    assert events[-1]["status"] == "success" and events[2]["history_stats"]["protocol"] == "fake-stateless"


def test_recorder_frame_content_and_smoke_title(tmp_path):
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is required for annotated MP4 recording")
    from gpt_field.agent_speedup_v1_video import AgentSpeedUpV1VideoRecorder, json_safe
    from agent_speedup_fakes import FakeWristSim
    wrist = FakeWristSim()
    obs = wrist.observation()
    recorder = AgentSpeedUpV1VideoRecorder(tmp_path / "smoke.mp4", episode_id="SMOKE", offline_smoke=True)
    recorder.observe(obs)
    diagnostics = {"step": 0, "controller": "agent_skill", "refreshed": True, "prediction": None,
                   "routing": {"skill": "grasp_with", "situation": "x" * 400,
                               "params": {"target": "black_bowl", "strategy": "rim_pinch", "side": "wrist_left",
                                          "approach": None, "waypoint_offset_m": None, "more_waypoints_m": None, "max_travel_m": None}},
                   "api_latency_s": np.float64(2.5), "api_calls_in_act": ["decision", "pointing_coarse"],
                   "macro": "grasp_with", "primitive": "align", "execution_kind": "agent_skill:align",
                   "macro_step": np.int64(3), "history_stats": {"calls_by_kind": {"decision": 1}},
                   "outcome": {"skill": "reach", "status": "failed", "reason": "budget",
                               "evidence": {"mark_uv": [100.0, 120.0], "setpoint_uv": [255.5, 280.0]}},
                   "pointing": [{"stage": "fine", "point_id": 28, "uv": [500.0, 20.0]}], "nan": float("nan")}
    recorder.decision(obs, Decision((0.0, 0.1, 0.0, 4e-9, 0.0, 0.0, 1.0), diagnostics), decision_latency_s=2.6)
    frame = recorder.render_frame()
    assert frame.size == (1600, 960)
    recorder.executed(wrist.step((0.0, 0.1, 0.0, 4e-9, 0.0, 0.0, 1.0)), step=0, success=False,
                      sim_status={"steps": 1}, feedback={"gap_mm": np.float32(70.0)}, history_stats={"protocol": "p"})
    recorder.close(status="diagnostic_complete")
    events = rows(tmp_path / "smoke.events.jsonl")
    assert [event["event"] for event in events] == ["observation", "skill_decision", "executed", "terminal"]
    assert events[1]["macro_step"] == 3 and events[1]["api_latency_s"] == 2.5 and events[1]["offline_smoke"]
    assert events[2]["feedback"] == {"gap_mm": 70.0} and events[2]["history_stats"] == {"protocol": "p"}
    assert json_safe({"a": (np.bool_(True), np.array([1.5, np.inf]))}) == {"a": [True, [1.5, None]]}
    # The skill panel never reads a model action chunk.
    assert recorder.chunk == []


# ---- real-module integration (skipped until Modules D/E exist) ----------------
def test_main_preflight_with_real_client(tmp_path, monkeypatch, capsys):
    client_module = pytest.importorskip("gpt_field.agent_speedup_v1_client")
    import gpt_field.chatgpt_auth as auth
    import gpt_field.chatgpt_plan as plan
    monkeypatch.setattr(auth, "SubscriptionCredentials", lambda *a, **k: pytest.fail("credentials read"))
    monkeypatch.setattr(plan, "_open_public", lambda *a, **k: pytest.fail("network used"))
    monkeypatch.setattr(runner, "SimProcess", lambda *a, **k: pytest.fail("simulator started"))
    output = tmp_path / "trial"
    assert runner.main(["--preflight", "--output", str(output)]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert not output.exists()
    payload = printed["payload"]
    assert payload["model"] == "gpt-6-astra" and payload["store"] is False and payload["stream"] is True
    assert payload["reasoning"] == {"effort": "medium"} and len(payload["input"]) == 1
    assert printed["client"]["prompt_version"] == client_module.AGENT_PROMPT_VERSION
    agent = printed["agent_speedup_v1"]
    assert agent["decision_schema_property_order"] == ["situation", "skill", "params"]
    assert agent["decision_schema_sha256"] and agent["pointing_schema_sha256"]["coarse"]
    assert set(printed["pointing_payloads"]) == {"coarse", "fine"}


def test_real_client_scripted_smoke_with_fake_policy(tmp_path):
    client_module = pytest.importorskip("gpt_field.agent_speedup_v1_client")
    from gpt_field.observation import decode_image, image_to_data_url

    def upright(url):
        rgb = np.asarray(decode_image(url))
        return image_to_data_url(np.ascontiguousarray(rgb[:, ::-1]), size=rgb.shape[0], rotate=False)

    class RealClientPolicy(FakePolicy):
        def act(self, observation):
            self.events.append("act")
            refreshed = self.index == 0
            routing = None
            if refreshed:
                choice = self.client.decide({"state_summary": {"eef_xyz_m": list(observation.state[:3])},
                                             "external_image": upright(observation.image),
                                             "wrist_image": upright(observation.wrist_image)})
                routing = {key: choice[key] for key in ("skill", "params", "situation")}
            self._pending = Decision((0.0, 0.0, 0.0, 4e-9, 0.0, 0.0, -1.0),
                {"step": self.index, "controller": "agent_skill", "refreshed": refreshed, "routing": routing,
                 "api_latency_s": None, "api_calls_in_act": ["decision"] if refreshed else [],
                 "execution_kind": "agent_skill:hold", "macro": None, "primitive": "hold", "macro_step": 0,
                 "update_steps": 1, "history_stats": self.client.history_stats(), "feedback": None,
                 "plan_refreshed": False, "outcome": None, "pointing": []})
            return self._pending

    output, result = run(tmp_path, smoke=True, smoke_steps=3, client_factory=None,
                         policy_factory=lambda client, **kw: RealClientPolicy(client, **kw), sim_factory=sim_factory())
    assert result["status"] == "diagnostic_complete", explain(output, result)
    assert result["billing"] == "offline_mock" and result["api_calls"] == 0 and result["mock_response_calls"] == 1
    assert list((output / "requests").glob("*.request.json"))


def test_real_policy_scripted_smoke_on_fake_simulator(tmp_path):
    pytest.importorskip("gpt_field.agent_speedup_v1_client")
    pytest.importorskip("gpt_field.agent_speedup_v1")
    output, result = run(tmp_path, smoke=True, smoke_steps=80, client_factory=None, policy_factory=None,
                         sim_factory=sim_factory(), recorder_factory=None, video=False)
    assert result["status"] == "diagnostic_complete", explain(output, result)
    assert result["steps"] == 80 and result["api_calls"] == 0
    rows(output / "trace.jsonl")
    assert json.loads((output / "agent_stats.json").read_text()) is not None


# ---- finalization robustness (R1), stop sources (R8), outcome reconciliation (R7a), stuck alert (R3) -------
def interrupting_policy(at=3):
    return lambda client, **kw: FakePolicy(client, fail_at=at, error=KeyboardInterrupt(), **kw)


class InterruptingCloseRecorder(FakeRecorder):
    """A second Ctrl+C that surfaces as KeyboardInterrupt inside recorder.close()."""

    def close(self, **kwargs):
        super().close(**kwargs)
        summary = self.path.parent / "summary.json"
        self.summary_at_close = json.loads(summary.read_text()) if summary.is_file() else None
        raise KeyboardInterrupt


def test_interrupt_during_video_finalization_keeps_summary_artifacts(tmp_path):
    output, result = run(tmp_path, policy_factory=interrupting_policy(), recorder_factory=InterruptingCloseRecorder)
    assert result["status"] == "user_stop" and result["steps"] == 3, explain(output, result)
    assert result["error"]["type"] == "KeyboardInterrupt" and result["error"]["source"] == "sigint"
    for name in FINAL_ARTIFACTS:
        assert (output / name).is_file(), name
    summary = json.loads((output / "summary.json").read_text())
    assert summary["status"] == "user_stop" and summary["finalization"]["stage"] == "complete"
    assert json.loads((output / "task_evaluation.json").read_text())["status"] == "user_stop"
    assert rows(output / "episodes.jsonl")[0]["status"] == "user_stop"
    assert json.loads((output / "ledger.json").read_text()) == []
    assert json.loads((output / "agent_stats.json").read_text())["acts"] == 4
    assert json.loads((output / "run_state.json").read_text())["stage"] == "finished"
    video_error = result["video_error"]
    assert video_error["type"] == "KeyboardInterrupt" and video_error["stage"] == "video_finalization"
    assert video_error["detail"] == "" and result["cleanup_errors"] == []
    # The pre-video summary was already on disk when the video was being finalized.
    before = FakeRecorder.instances[-1].summary_at_close
    assert before["status"] == "user_stop" and before["finalization"]["stage"] == "pre_video_finalization"
    stop = [row for row in rows(output / "trace.jsonl") if row["event"] == "operator_stop"]
    assert stop == [{"event": "operator_stop", "source": "sigint", "step": 3, "feedback_count": 3,
                     "pending_action_index": None, "request_file": None, "stage": "deciding"}]
    assert "KeyboardInterrupt at stage deciding" in (output / "error_traceback.txt").read_text()


def test_video_error_detail_is_redacted_and_never_overrides_success(tmp_path):
    class FailingRecorder(FakeRecorder):
        def close(self, **kwargs):
            raise RuntimeError(f"ffmpeg failed: Authorization: Bearer {LEAK} " + "x" * 900)

    output, result = run(tmp_path, recorder_factory=FailingRecorder)
    assert result["status"] == "success" and result["evaluation_completed"]
    detail = result["video_error"]["detail"]
    assert len(detail) == runner.ERROR_DETAIL_CHARS and "ffmpeg failed" in detail and LEAK not in detail
    assert result["video_error"]["message"] == "Diagnostic stopped; no retry or API-key fallback"
    assert LEAK not in (output / "summary.json").read_text()


def test_deferred_signals_during_finalization(tmp_path, capsys):
    class SignallingRecorder(FakeRecorder):
        def close(self, **kwargs):
            super().close(**kwargs)
            os.kill(os.getpid(), signal.SIGINT)   # second Ctrl+C: deferred, never raised

    previous = signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM)
    output, result = run(tmp_path, policy_factory=interrupting_policy(), recorder_factory=SignallingRecorder)
    assert result["status"] == "user_stop" and result["video_error"] is None, explain(output, result)
    assert result["finalization"] == {"stage": "complete", "deferred_signals": ["SIGINT"]}
    assert (signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM)) == previous
    err = capsys.readouterr().err
    assert "finalizing: writing summary/video; press Ctrl+C 2 more times to abort" in err
    assert "press Ctrl+C 1 more time to abort" in err
    for name in FINAL_ARTIFACTS:
        assert (output / name).is_file(), name


def test_third_press_writes_pre_video_summary_then_hard_exits(tmp_path, monkeypatch, capsys):
    output = tmp_path / "trial"
    exits = []

    def hard_exit(code):
        summary = output / "summary.json"
        exits.append((code, json.loads(summary.read_text())["finalization"]["stage"] if summary.is_file() else None,
                      json.loads((output / "run_state.json").read_text())["stage"]))

    class SignallingCloseSim(FakeLiberoSim):
        def close(self):
            super().close()
            os.kill(os.getpid(), signal.SIGINT)    # press 2
            os.kill(os.getpid(), signal.SIGTERM)   # press 3 (SIGTERM counts too): abort

    monkeypatch.setattr(runner, "_hard_exit", hard_exit)
    previous = signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM)
    _, result = run(tmp_path, policy_factory=interrupting_policy(2), sim_factory=lambda c: SignallingCloseSim(c))
    assert exits == [(130, "aborted_before_video_finalization", "finalization_aborted")]
    assert result["status"] == "user_stop" and result["cleanup_errors"] == []
    assert result["finalization"]["deferred_signals"] == ["SIGINT", "SIGTERM"]
    assert (signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM)) == previous
    assert "finalizing aborted by operator" in capsys.readouterr().err


@pytest.mark.parametrize("close_error", [RuntimeError(f"worker hung Bearer {LEAK}"), KeyboardInterrupt()])
def test_simulator_close_errors_never_override_outcome_statuses(tmp_path, close_error):
    class FailingCloseSim(FakeLiberoSim):
        def close(self):
            super().close()
            raise close_error

    _, result = run(tmp_path, name="success", sim_factory=lambda c: FailingCloseSim(c, success_at=4))
    assert result["status"] == "success" and result["evaluation_completed"]
    [record] = result["cleanup_errors"]
    assert record["stage"] == "simulator_close" and record["type"] == type(close_error).__name__
    assert LEAK not in json.dumps(result)
    output, result = run(tmp_path, name="stop", policy_factory=interrupting_policy(),
                         sim_factory=lambda c: FailingCloseSim(c))
    assert result["status"] == "user_stop" and result["cleanup_errors"][0]["stage"] == "simulator_close"
    assert json.loads((output / "summary.json").read_text())["status"] == "user_stop"
    _, result = run(tmp_path, name="smoke", smoke=True, smoke_steps=3, transport=OfflineTransport(),
                    sim_factory=lambda c: FailingCloseSim(c))
    assert result["status"] == "diagnostic_complete" and result["diagnostic_complete"]
    api = lambda client, **kw: FakePolicy(client, fail_at=1, error=APIError("quota"), **kw)
    _, result = run(tmp_path, name="api", policy_factory=api, sim_factory=lambda c: FailingCloseSim(c))
    assert result["status"] == "infrastructure_error" and result["error"]["type"] == "APIError"


class OutcomePolicy(FakePolicy):
    """FakePolicy with numbered outcomes: script[i] is the outcome list reported by act i."""

    def __init__(self, client, *, script=(), late=None, **kwargs):
        super().__init__(client, **kwargs)
        self.script, self.late, self.outcomes, self.counter = list(script), late, [], 0

    def reset(self):
        super().reset()
        self.outcomes, self.counter = [], 0

    def _numbered(self, outcome):
        self.counter += 1
        outcome = {**outcome, "n": self.counter, "step": self.index}
        self.outcomes = (self.outcomes + [outcome])[-50:]
        return outcome

    def act(self, observation):
        if self.late is not None and self.index == self.late[0]:
            self._numbered(self.late[1])   # finished inside the act the interrupt aborts
            raise KeyboardInterrupt
        decision = super().act(observation)
        reported = [self._numbered(item) for item in (self.script[self.index] if self.index < len(self.script) else [])]
        details = {**decision.diagnostics, "outcomes": reported, "outcome": reported[-1] if reported else None}
        self._pending = Decision(decision.action, details)
        return self._pending


def test_outcome_missing_from_acts_is_reconciled_at_finalization(tmp_path):
    calibrated = {"skill": "calibrate", "status": "succeeded", "reason": "calibrated"}
    late = {"skill": "grasp_with", "status": "failed", "reason": "budget"}
    policy = lambda client, **kw: OutcomePolicy(client, script=[[calibrated]], late=(3, late), **kw)
    output, result = run(tmp_path, policy_factory=policy)
    assert result["status"] == "user_stop" and result["steps"] == 3
    assert result["macro_outcome_sequence"] == [
        {"n": 1, "step": 0, "skill": "calibrate", "status": "succeeded", "reason": "calibrated"},
        {"n": 2, "step": 3, "skill": "grasp_with", "status": "failed", "reason": "budget",
         "recorded_at_finalization": True}]
    assert result["macro_outcomes"] == 2 and result["outcomes_recorded_at_finalization"] == 1
    assert result["macro_outcome_counts"] == {"calibrate:succeeded": 1, "grasp_with:failed": 1}
    trace = rows(output / "trace.jsonl")
    recorded = [row for row in trace if row["event"] == "macro_outcome"]
    assert [row["outcome"]["n"] for row in recorded] == [1, 2]
    assert recorded[-1]["recorded_at_finalization"] is True and "recorded_at_finalization" not in recorded[0]
    assert [row["event"] for row in trace[-2:]] == ["macro_outcome", "operator_stop"]
    assert result["cleanup_errors"] == []


def test_stuck_alert_is_operator_facing_only(tmp_path, capsys):
    failed = {"skill": "reach", "status": "failed", "reason": "budget"}
    policy = lambda client, **kw: OutcomePolicy(client, script=[[failed]] * 13, **kw)
    output, result = run(tmp_path, policy_factory=policy, sim_factory=sim_factory(success_at=13))
    assert result["status"] == "success" and result["steps"] == 13 and result["macro_outcomes"] == 13
    alert = json.loads((output / "run_state.json").read_text())["stuck_alert"]
    stop = f"touch {output / 'operator_stop.request.json'}"
    assert alert["active"] is True and alert["automatic_cap"] is False and alert["outcome_n"] == 10
    assert alert["reason"] == "10 of the last 10 macro outcomes failed or were rejected"
    assert alert["step"] == 9 and alert["stop_command"] == stop and alert["api_calls"] >= 1
    assert isinstance(alert["wall_s"], float) and result["stuck_alerts"] == [alert]
    captured = capsys.readouterr()
    [line] = [line for line in captured.err.splitlines() if line.startswith("STUCK ALERT")]
    assert "step 9" in line and "api calls" in line and "wall" in line and line.endswith(stop)
    assert captured.err.count(stop) == 2   # startup + alert
    console = [json.loads(line) for line in captured.out.splitlines() if line.startswith("{")]
    outcomes = [row for row in console if row["event"] == "outcome"]
    assert [row["n"] for row in outcomes] == list(range(1, 14))
    assert set(outcomes[0]) == {"event", "n", "skill", "status", "reason", "step", "wall_s", "api_calls"}
    assert [row["k"] for row in console if row["event"] == "decision"] == [1, 2, 3, 4]
    executed = [row for row in console if row["event"] == "executed"]
    assert len(executed) == 1 and executed[0]["steps"] == 13 and executed[0]["success_observed"]


def test_stuck_alert_rearms_after_success_and_long_runs_print_every_25th_step(tmp_path, capsys):
    failed = {"skill": "reach", "status": "failed", "reason": "budget"}
    succeeded = {"skill": "reach", "status": "succeeded", "reason": "goal_reached"}
    script = [[failed]] * 10 + [[succeeded]] + [[failed]] * 3
    policy = lambda client, **kw: OutcomePolicy(client, script=script, **kw)
    output, result = run(tmp_path, policy_factory=policy, sim_factory=sim_factory(success_at=None),
                         smoke=True, smoke_steps=51, transport=OfflineTransport())
    assert result["status"] == "diagnostic_complete"
    first, second = result["stuck_alerts"]
    assert first["active"] is False and first["cleared_by_outcome_n"] == 11 and second["outcome_n"] == 12
    captured = capsys.readouterr()
    assert captured.err.count("STUCK ALERT") == 2
    executed = [json.loads(line) for line in captured.out.splitlines() if line.startswith('{"event": "executed"')]
    assert [row["steps"] for row in executed] == [25, 50, 51] and executed[-1]["final"] is True


def test_stuck_reason_rules():
    out = lambda skill, status, reason="r": {"skill": skill, "status": status, "reason": reason, "n": None}
    assert runner._stuck_reason([out("reach", "failed")] * 9) is None
    assert runner._stuck_reason([out("reach", "rejected")] * 10).startswith("10 of the last 10")
    assert runner._stuck_reason([out("reach", "succeeded")] * 3 + [out("reach", "failed")] * 9) is None
    lost = [out("grasp_with", "succeeded"), out("transport", "failed", "lost_in_transport")]
    assert runner._stuck_reason(lost + [out("grasp_with", "failed")] * 5) is None
    assert "followed by 6 failed grasp_with" in runner._stuck_reason(
        lost + [out("grasp_with", "failed"), out("reach", "succeeded")] * 6)
    assert runner._stuck_reason(lost + [out("grasp_with", "failed")] * 6 + [out("grasp_with", "succeeded")]) is None


def test_external_axes_cli_and_policy_factory(tmp_path, monkeypatch):
    agent_policy = pytest.importorskip("gpt_field.agent_speedup_v1")
    assert runner.EXTERNAL_AXES_MODES == agent_policy.EXTERNAL_AXES_MODES
    captured = []

    def fake_execute(output, **kwargs):
        captured.append(kwargs)
        return {"evaluation_completed": False, "diagnostic_complete": True}

    monkeypatch.setattr(runner, "execute", fake_execute)
    assert runner.main(["--output", str(tmp_path / "a"), "--sim-smoke", "--external-axes", "off"]) == 0
    assert runner.main(["--output", str(tmp_path / "b"), "--sim-smoke"]) == 0
    assert [kwargs["external_axes"] for kwargs in captured] == ["off", "measured"]
    with pytest.raises(SystemExit):
        runner.main(["--output", str(tmp_path / "c"), "--sim-smoke", "--external-axes", "sideways"])
    monkeypatch.undo()

    seen = []

    def default_factory(client, *, local_retries=2, external_axes="measured"):
        seen.append(external_axes)
        return FakePolicy(client, local_retries=local_retries)

    monkeypatch.setattr(runner, "_default_policy_factory", default_factory)
    output, result = run(tmp_path, name="off", policy_factory=None, external_axes="off")
    assert seen == ["off"] and result["status"] == "success" and result["external_axes"] == "off"
    identity = json.loads((output / "manifest.json").read_text())["configuration"]
    assert identity["external_axes"] == "off" and identity["policy"]["external_axes"] == "off"
    with pytest.raises(ValueError, match="external_axes"):
        run(tmp_path, name="bad", external_axes="sideways")

    def mismatched(client, **kw):
        policy = FakePolicy(client, **kw)
        policy.external_axes = "measured"
        return policy

    with pytest.raises(ValueError, match="differs"):
        run(tmp_path, name="mismatch", policy_factory=mismatched, external_axes="off")
    assert not (tmp_path / "bad").exists() and not (tmp_path / "mismatch").exists()
    preview = runner.preflight(CONFIG, runner.build_sim_config("bowl_plate"), client_factory=FakeClient,
                               external_axes="off")
    assert preview["external_axes"] == "off"
    with pytest.raises(ValueError):
        runner.preflight(CONFIG, runner.build_sim_config("bowl_plate"), client_factory=FakeClient,
                         external_axes="sideways")


_ECHO_WORKER = r"""
import json, os, sys
for line in sys.stdin:
    request = json.loads(line)
    big = "x" * 400000 if request["op"] == "tasks" else ""
    print(json.dumps({"ok": True, "result": {"sid": os.getsid(0), "pid": os.getpid(), "big": big}}), flush=True)
"""


def test_session_sim_process_isolates_and_reaps_worker():
    sim = runner._SessionSimProcess(runner.build_sim_config("bowl_plate"), command=[sys.executable, "-c", _ECHO_WORKER])
    sim.start()
    worker = sim._process
    assert sim.metadata["sid"] == sim.metadata["pid"] == worker.pid and worker.pid != os.getsid(0)
    # An in-flight reply larger than the pipe buffer (a worker mid-step at Ctrl+C) must not wedge close().
    worker.stdin.write(b'{"op": "tasks"}\n')
    began = time.monotonic()
    sim.close()
    assert time.monotonic() - began < 2.5 and worker.poll() == 0 and sim._process is None


def test_session_sim_process_launches_like_simprocess(monkeypatch):
    calls = []

    class Launched(Exception):
        pass

    def fake_popen(command, **kwargs):
        calls.append((list(command), kwargs))
        raise Launched

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    config = runner.build_sim_config("bowl_plate")
    for kind in (runner.SimProcess, runner._SessionSimProcess):
        with pytest.raises(Launched):
            kind(config).start()
    (base_command, base), (session_command, session) = calls
    assert session_command == base_command and session.pop("start_new_session") is True
    assert "start_new_session" not in base and session.pop("env") == base.pop("env") and session == base


def test_video_recorder_builds_session_writer_without_patching(tmp_path):
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is required for annotated MP4 recording")
    import gpt_field.video as video
    from gpt_field.agent_speedup_v1_video import AgentSpeedUpV1VideoRecorder, _SessionMP4Stream
    from gpt_field.direct_history_video import DirectHistoryVideoRecorder

    popen, stream = subprocess.Popen, video.MP4Stream
    base = DirectHistoryVideoRecorder(tmp_path / "base.mp4", episode_id="E", offline_smoke=True)
    agent = AgentSpeedUpV1VideoRecorder(tmp_path / "agent.mp4", episode_id="E", offline_smoke=True)
    try:
        assert subprocess.Popen is popen and video.MP4Stream is stream
        assert set(vars(base)) <= set(vars(agent))
        for key in set(vars(base)) - {"writer", "_events", "_started", "path", "events_path"}:
            assert getattr(agent, key) == getattr(base, key), key
        assert type(agent.writer) is _SessionMP4Stream and type(base.writer) is video.MP4Stream
        assert set(vars(agent.writer)) == set(vars(base.writer))
        assert agent.writer._process.args[:-1] == base.writer._process.args[:-1]
        assert os.getsid(agent.writer._process.pid) == agent.writer._process.pid
        assert os.getsid(base.writer._process.pid) == os.getsid(0)
    finally:
        base.close(status="diagnostic_complete")
        agent.close(status="diagnostic_complete")


def test_real_recorder_survives_second_interrupt_during_close(tmp_path):
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is required for annotated MP4 recording")
    from gpt_field.agent_speedup_v1_video import AgentSpeedUpV1VideoRecorder

    class SecondInterrupt(AgentSpeedUpV1VideoRecorder):
        def close(self, **kwargs):
            os.kill(os.getpid(), signal.SIGINT)
            return super().close(**kwargs)

    output, result = run(tmp_path, policy_factory=interrupting_policy(), recorder_factory=SecondInterrupt)
    assert result["status"] == "user_stop" and result["video"] == "episode.mp4", explain(output, result)
    assert result["video_error"] is None and result["video_partial"] is None
    assert result["finalization"]["deferred_signals"] == ["SIGINT"]
    assert (output / "episode.mp4").stat().st_size > 1000
    assert rows(output / "episode.events.jsonl")[-1]["status"] == "user_stop"


def test_sigterm_stop_is_a_user_stop_with_sigterm_source(tmp_path):
    class TerminatedPolicy(FakePolicy):
        def act(self, observation):
            if self.index == 2:
                os.kill(os.getpid(), signal.SIGTERM)   # termination_as_interrupt -> KeyboardInterrupt
            return super().act(observation)

    output, result = run(tmp_path, policy_factory=lambda client, **kw: TerminatedPolicy(client, **kw))
    assert result["status"] == "user_stop" and result["steps"] == 2 and result["error"]["source"] == "sigterm"
    stop = [row for row in rows(output / "trace.jsonl") if row["event"] == "operator_stop"]
    assert [(row["source"], row["stage"]) for row in stop] == [("sigterm", "deciding")]
    for name in FINAL_ARTIFACTS:
        assert (output / name).is_file(), name
