"""Opt-in speed-first execution uses mock subscription and simulator only."""
from dataclasses import replace
import hashlib
import json

import pytest

from gpt_field.distance_tasks import resolve_distance_task
from test_distance_policy import intent
from test_distance_runner import SETTINGS
from test_field_history import FakeTransport, envelope
from test_field_history_runner import FakeRecorder, fake_simulator, observation, rows
from test_hybrid_subscription_runner import runner, sim_config


OPTIONS = {**SETTINGS, "speed_first": True}


def monitored_simulator(obs, *, task="bowl_plate", success_at=2, collision_at=1,
                        output=None, request_stop_at=None, bad_monitor=None):
    base, instances = fake_simulator(obs, success_at=success_at)
    spec = resolve_distance_task(task)

    class Simulator(base):
        def list_tasks(self):
            return [{"task_id": spec.task_id, "prompt": spec.prompt}]

        def _status(self):
            detected = collision_at is not None and len(self.actions) >= collision_at
            self.status.update(libero_success=self.status["success"], collision={
                "enabled": True, "mode": "task_manipulation_v1", "detected": detected,
                "stop_on_collision": False, "collision_policy": "record_only",
                "success_qualification": "raw_libero_goal",
                "first_event": {"private_contact": "CONTACT_MUST_NOT_ENTER_POLICY"} if detected else None})
            if bad_monitor == "missing":
                del self.status["collision"]
            elif bad_monitor == "strict":
                self.status["collision"]["stop_on_collision"] = True
            elif bad_monitor == "qualified":
                self.status["success"] = self.status["libero_success"] and not detected
            elif bad_monitor == "collision_termination":
                self.status["terminated"] = detected

        def reset(self, task_id, episode_index):
            assert (task_id, episode_index) == (spec.task_id, 0)
            result = super().reset(0, 0)
            self._status()
            if request_stop_at == 0:
                (output / "operator_stop.request.json").write_text('{"reason":"fixture"}')
            return result

        def step(self, action):
            obs, _ = super().step(action)
            self._status()
            if len(self.actions) == request_stop_at:
                (output / "operator_stop.request.json").write_text('{"reason":"fixture"}')
            return obs, self.status["success"]

        def close(self):
            super().close()
            self.status = {}  # A real SimProcess also clears status during cleanup.

    return Simulator, instances


def run(output, obs, *, task="bowl_plate", success_at=2, collision_at=1,
        request_stop_at=None, bad_monitor=None, responses=None, **kwargs):
    spec = resolve_distance_task(task)
    obs = replace(obs, prompt=spec.prompt)
    simulator, instances = monitored_simulator(obs, task=task, success_at=success_at,
        collision_at=collision_at, output=output, request_stop_at=request_stop_at, bad_monitor=bad_monitor)
    responses = responses if responses is not None else [intent(direct=True, phase="grasp")]
    transport = FakeTransport(*(envelope(raw, id=f"fixture-{i}", model="gpt-6-astra",
        reasoning={"effort": "medium"}) for i, raw in enumerate(responses)))
    result = runner.execute_episode(output, config=runner.configuration(),
        sim_config=sim_config(suite=spec.suite, collision_check_mode="task_manipulation_v1", stop_on_collision=False),
        task_key=task, token_provider=lambda: "offline-oauth", transport=transport,
        sim_factory=simulator, recorder_factory=FakeRecorder, **OPTIONS, **kwargs)
    return result, transport, instances


@pytest.mark.parametrize("task", ["bowl_plate", "drawer_bowl"])
def test_cli_preset_is_subscription_zero_shot_record_only_and_has_no_caps(task, capsys):
    assert runner.main(["--speed-first", "--task", task, "--preflight"]) == 0
    p = json.loads(capsys.readouterr().out)
    assert p["speed_first"] and p["zero_shot"] and "demonstration" not in p
    assert p["payload"]["reasoning"] == {"effort": "medium"}
    assert p["payload"]["model"] == "gpt-6-astra"
    assert p["simulator"]["collision_check_mode"] == "task_manipulation_v1"
    assert p["simulator"]["stop_on_collision"] is False
    assert p["policy"]["method"] == "speed_first_distance_hybrid"
    assert p["policy"]["field_update_steps"] == p["policy"]["field_reuse_limit"] == 30
    assert p["policy"]["direct_action_steps"] == 5 and p["policy"]["planner"]["horizon"] == 10
    assert p["policy"]["field_refresh_policy"] == "events_or_30_actions"
    assert not p["policy"]["pixel_goal_control"] and not p["policy"]["automatic_view_recovery"]
    assert p["collision_policy"] == "record_only" and p["success_qualification"] == "raw_libero_goal"
    assert p["primary_timing_metric"] == "control_without_video_s"
    assert p["max_policy_steps"] is p["max_api_calls"] is p["max_wall_time_s"] is None
    assert p["api_calls"] == p["api_key_calls"] == 0
    assert not any(p[k] for k in ("credentials_checked", "simulator_started", "files_written", "stop_on_model_complete"))
    assert p["stopping_rule"] == runner.STOPPING_RULE


@pytest.mark.parametrize("flags", [
    ["--reasoning-effort", "high"], ["--execution-mode", "fixed"], ["--feedback-history", "full"],
    ["--field-representation", "dense"], ["--controller-mode", "direct"],
    ["--zero-shot-collision-free"], ["--historical-zero-shot-direct"], ["--timing-comparison"],
    ["--offline-smoke"], ["--stop-on-model-complete"], ["--demonstration", "absent.json"],
])
def test_cli_rejects_incompatible_explicit_settings_before_side_effects(flags, tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "configuration", lambda: pytest.fail("Invalid options reached configuration"))
    output = tmp_path / "run"
    with pytest.raises(SystemExit) as exc:
        runner.main(["--speed-first", "--output", str(output), *flags])
    assert exc.value.code == 2 and not output.exists()


@pytest.mark.parametrize("task", ["bowl_plate", "drawer_bowl"])
@pytest.mark.parametrize("collision_at", [None, 0, 1, 2])
def test_collisions_are_secondary_and_do_not_stop_or_disqualify_raw_libero_success(
        tmp_path, observation, task, collision_at):
    output = tmp_path / "run"
    result, transport, instances = run(output, observation, task=task, collision_at=collision_at)
    assert result["status"] == "success", result["error"]
    assert result["steps"] == result["measured_feedback_count"] == 2
    assert result["libero_success"] is result["task_success"] is result["completion_time_eligible"] is True
    assert result["collision"]["detected"] is (collision_at is not None)
    assert result["failure_reason"] is None
    assert result["evaluation_completed"] and result["success_rate"] == 1. and result["scored_episodes"] == 1
    assert result["control_without_video_s"] == pytest.approx(
        result["control_elapsed_s"] - result["timing_components"]["video_render_s"])
    assert result["control_without_video_s"] > 0 and result["api_calls"] == 1 and result["api_key_calls"] == 0
    assert instances[0].closed and len(instances[0].actions) == 2
    assert instances[0].actions == intent(direct=True, phase="grasp")["direct"]["actions"][:2]
    assert json.loads((output / "task_evaluation.json").read_text())["completion_time_eligible"] is True
    records = rows(output / "trace.jsonl")
    assert not any(r["event"] == "forbidden_collision" for r in records)
    assert sum(r["event"] == "recorded_collision" for r in records) == int(collision_at in (1, 2))
    for marker in ("CONTACT_MUST_NOT_ENTER_POLICY", "MUST_NOT_ENTER_POLICY", "external_demonstration"):
        assert marker not in json.dumps(transport.calls)
    manifest = json.loads((output / "manifest.json").read_text())["configuration"]
    assert manifest["policy"]["class"] == "gpt_field.speed_first_hybrid.SpeedFirstHybridPolicy"
    assert manifest["mode"] == "live_chatgpt_subscription_hybrid"  # Legacy recording metadata remains readable.
    sealed = output / "frozen_source/speed_first_hybrid.py"
    assert manifest["software"]["source_sha256"][sealed.name] == hashlib.sha256(sealed.read_bytes()).hexdigest()


@pytest.mark.parametrize("request_stop_at", [0, 1])
def test_operator_stop_is_at_feedback_boundary_and_is_not_completion(tmp_path, observation, request_stop_at):
    output = tmp_path / "run"
    result, transport, instances = run(output, observation, success_at=None, request_stop_at=request_stop_at)
    assert result["status"] == "user_stop" and not result["libero_success"]
    assert result["steps"] == result["measured_feedback_count"] == request_stop_at
    assert result["pending_action_index"] is None
    assert not result["completion_time_eligible"] and not result["task_success"]
    assert result["incomplete_episodes"] == 1 and result["scored_episodes"] == 0
    assert result["success_rate"] is None and result["failure_reason"] is None
    assert len(transport.calls) == int(request_stop_at > 0) and instances[0].closed
    records = rows(output / "trace.jsonl")
    assert records[-1]["event"] == "operator_stop"
    if request_stop_at:
        assert records[-2]["event"] == "feedback"


@pytest.mark.parametrize("bad_monitor", ["missing", "strict", "qualified", "collision_termination"])
def test_bad_collision_contract_cannot_be_silently_scored(tmp_path, observation, bad_monitor):
    result, _, instances = run(tmp_path / "run", observation, bad_monitor=bad_monitor, collision_at=1,
        success_at=1 if bad_monitor == "qualified" else 2)
    assert result["status"] == "infrastructure_error"
    assert not result["completion_time_eligible"] and not result["evaluation_completed"]
    assert instances[0].closed


def test_speed_first_field_then_five_direct_actions_with_measured_feedback(tmp_path, observation):
    result, transport, instances = run(tmp_path / "run", observation, success_at=6, collision_at=1,
        responses=[intent(goal=(.03, 0., 0.)), intent(direct=True, phase="grasp")])
    assert result["status"] == "success", result["error"]
    assert result["coarse_field_actions"] == 1 and result["contact_direct_actions"] == 5
    assert result["measured_feedback_count"] == 6 and result["controller_switches"] == 1
    assert len(transport.calls) == 2
    assert instances[0].actions[1:] == intent(direct=True, phase="grasp")["direct"]["actions"]
    records = rows(tmp_path / "run/trace.jsonl")
    first = next(r for r in records if r["event"] == "action")
    assert first["diagnostics"]["plan"]["search"]["method"] == "dijkstra_distance_field"
    assert not any(r["event"] == "local_visual_feedback" for r in records)


def test_programmatic_speed_first_rejects_unmonitored_or_strict_config():
    for config in (sim_config(), sim_config(collision_check_mode="task_manipulation_v1")):
        with pytest.raises(ValueError, match="record-only"):
            runner.preflight(runner.configuration(), config, **OPTIONS)
