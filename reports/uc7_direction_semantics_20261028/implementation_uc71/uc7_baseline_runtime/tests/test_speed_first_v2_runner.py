"""V2 runner integration: mock subscription transport and mock robot only."""
from dataclasses import replace
import hashlib
import json

import pytest

from gpt_field.speed_first_v2 import ChatGPTPlanSpeedFirstV2Client, SPEED_FIRST_V2_VERSION
from test_distance_policy import intent
from test_distance_runner import SETTINGS
from test_field_history import FakeTransport, envelope
from test_field_history_runner import FakeRecorder, observation, rows
from test_hybrid_subscription_runner import runner, sim_config
from test_speed_first_runner import monitored_simulator


OPTIONS = {**SETTINGS, "speed_first": True, "speed_first_v2": True}


def v2_intent(*, operation=None, **kwargs):
    raw = intent(**kwargs)
    raw["gripper_operation"] = None if operation is None else {"command": operation}
    if operation is not None:
        raw.update(controller="contact_direct", near_target=True, field=None, direct=None)
    return raw


def run(output, obs, responses, *, success_at=None, collision_at=1,
        request_stop_at=None, move_fingers=True, clock=None, recorder=FakeRecorder):
    base, instances = monitored_simulator(obs, success_at=success_at,
        collision_at=collision_at, output=output, request_stop_at=request_stop_at)

    class Simulator(base):
        def step(self, action):
            measured, success = super().step(action)
            if move_fingers and action[6] != 0:
                # A deterministic six-step closure/opening followed by stable
                # measured fingers; no object attachment or success is inferred.
                state = list(measured.state)
                target = .016 if action[6] > 0 else .04
                magnitude = abs(state[6])
                magnitude += max(-.004, min(.004, target - magnitude))
                state[6:8] = [magnitude, -magnitude]
                self.obs = replace(measured, state=tuple(state))
            if clock is not None:
                clock[0] += .05
            return self.obs, success

    fake = FakeTransport(*(envelope(raw, id=f"v2-fixture-{index}", model="gpt-6-astra",
        reasoning={"effort": "medium"}) for index, raw in enumerate(responses)))

    def transport(*args):
        if clock is not None:
            clock[0] += 5.
        return fake(*args)

    result = runner.execute_episode(output, config=runner.configuration(),
        sim_config=sim_config(collision_check_mode="task_manipulation_v1", stop_on_collision=False),
        token_provider=lambda: "offline-v2-oauth", transport=transport,
        sim_factory=Simulator, recorder_factory=recorder, **OPTIONS)
    return result, fake, instances


def text_records(payload):
    result = []
    for item in payload["input"]:
        for part in item.get("content", []):
            if part.get("type") in ("input_text", "output_text"):
                try:
                    value = json.loads(part["text"])
                except (ValueError, TypeError):
                    continue
                if isinstance(value, dict):
                    result.append(value)
    return result


def test_v2_preflight_uses_exact_client_schema_and_no_side_effects(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(runner, "SimProcess", lambda *a: pytest.fail("preflight started simulator"))
    monkeypatch.setattr(ChatGPTPlanSpeedFirstV2Client, "preflight",
                        lambda *a, **k: pytest.fail("preflight checked credentials"))
    monkeypatch.setattr(ChatGPTPlanSpeedFirstV2Client, "predict",
                        lambda *a, **k: pytest.fail("preflight called model"))
    output = tmp_path / "absent"
    assert runner.main(["--speed-first-v2", "--preflight", "--output", str(output)]) == 0
    record = json.loads(capsys.readouterr().out)
    assert not output.exists()
    assert record["speed_first"] and record["speed_first_v2"] and record["zero_shot"]
    assert record["speed_first_protocol"] == record["policy"]["prompt_version"] == SPEED_FIRST_V2_VERSION
    assert record["payload"]["text"]["format"]["name"] == "astra_speed_first_distance_hybrid_v2"
    schema = record["payload"]["text"]["format"]["schema"]
    assert "gripper_operation" in schema["required"]
    assert schema["properties"]["gripper_operation"]["anyOf"][0]["properties"]["command"]["enum"] == ["open", "close"]
    assert record["payload"]["model"] == "gpt-6-astra"
    assert record["payload"]["reasoning"] == {"effort": "medium"}
    assert record["billing"] == "chatgpt_subscription" and record["api_key_fallback"] is False
    assert record["simulator"]["seed"] == 7 and record["task_id"] == record["episode_index"] == 0
    assert record["simulator"]["stop_on_collision"] is False
    assert record["simulator"]["stop_on_success"] is True
    assert record["success_qualification"] == "raw_libero_goal"
    assert not record["stop_on_model_complete"]
    assert record["policy"]["direct_action_steps"] == 5
    assert record["policy"]["field_update_steps"] == record["policy"]["field_reuse_limit"] == 30
    assert record["max_policy_steps"] is record["max_api_calls"] is record["max_wall_time_s"] is None
    assert record["api_calls"] == record["api_key_calls"] == 0
    assert not any(record[k] for k in ("credentials_checked", "simulator_started", "files_written"))


@pytest.mark.parametrize("flags", [
    ["--reasoning-effort", "high"], ["--controller-mode", "direct"],
    ["--execution-mode", "fixed"], ["--feedback-history", "full"],
    ["--field-representation", "dense"], ["--demonstration", "missing.json"],
    ["--stop-on-model-complete"], ["--offline-smoke"], ["--timing-comparison"],
    ["--zero-shot-collision-free"], ["--historical-zero-shot-direct"],
])
def test_v2_cli_rejects_incompatible_protocol_before_output_or_credentials(flags, tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "configuration", lambda: pytest.fail("invalid arguments reached configuration"))
    output = tmp_path / "absent"
    with pytest.raises(SystemExit) as caught:
        runner.main(["--speed-first-v2", "--output", str(output), *flags])
    assert caught.value.code == 2 and not output.exists()


@pytest.mark.parametrize("success_at", [1, 7, 15])
def test_libero_success_ends_even_inside_local_operation_and_collision_is_secondary(
        tmp_path, observation, success_at):
    result, fake, instances = run(tmp_path / "trial", observation,
        [v2_intent(operation="close", phase="grasp")], success_at=success_at)
    assert result["status"] == "success", result["error"]
    assert result["steps"] == result["measured_feedback_count"] == success_at
    assert result["libero_success"] is result["task_success"] is result["completion_time_eligible"] is True
    assert result["collision"]["detected"] and result["failure_reason"] is None
    assert result["api_calls"] == len(fake.calls) == 1 and result["api_key_calls"] == 0
    assert instances[0].closed
    assert len(instances[0].actions) == success_at
    manifest = json.loads((tmp_path / "trial/manifest.json").read_text())["configuration"]
    assert manifest["policy"]["class"] == "gpt_field.speed_first_v2.SpeedFirstV2Policy"
    for name in ("speed_first_v2.py", "speed_first_gripper.py", "speed_first_history.py"):
        frozen = tmp_path / "trial/frozen_source" / name
        assert manifest["software"]["source_sha256"][name] == hashlib.sha256(frozen.read_bytes()).hexdigest()
    actions = [r for r in rows(tmp_path / "trial/trace.jsonl") if r["event"] == "action"]
    assert all(r["diagnostics"]["execution_kind"] == "local_gripper_settle" for r in actions)
    assert all(r["diagnostics"]["local_gripper"]["task_success_certified"] is False for r in actions)
    assert "CONTACT_MUST_NOT_ENTER_POLICY" not in json.dumps(fake.calls)
    assert "MUST_NOT_ENTER_POLICY" not in json.dumps(fake.calls)


def test_fifteen_local_close_actions_then_unchanged_direct5_and_no_model_in_between(tmp_path, observation):
    close = v2_intent(operation="close", phase="grasp")
    direct = v2_intent(direct=True, phase="verify")
    direct["direct"]["actions"] = [[.01 * i, 0., 0., 0., 0., 0., 1.] for i in range(1, 6)]
    output = tmp_path / "trial"
    result, fake, instances = run(output, observation, [close, direct], success_at=20)
    assert result["status"] == "success", result["error"]
    assert result["steps"] == result["measured_feedback_count"] == 20
    assert len(fake.calls) == result["api_calls"] == 2
    assert instances[0].actions[15:] == direct["direct"]["actions"]
    trace = rows(output / "trace.jsonl")
    actions = [r for r in trace if r["event"] == "action"]
    assert [r["step"] for r in actions if r["diagnostics"]["refreshed"]] == [0, 15]
    assert all(r["diagnostics"]["execution_kind"] == "local_gripper_settle" for r in actions[:15])
    assert all(r["diagnostics"]["execution_kind"] == "direct_five_actions" for r in actions[15:])
    assert [r["step"] for r in trace if r["event"] == "feedback"] == list(range(20))
    second = text_records(fake.calls[1])
    completed = next(r for r in second if r.get("type") == "local_gripper_result")
    assert completed["actions_executed"] == 15 and completed["stop_reason"] == "measured_fingers_stable"
    assert completed["grasp_success_certified"] is completed["task_success_certified"] is False
    feedback = next(r for r in second if r.get("type") == "execution_block_summary")
    assert feedback["executed_actions"] == 15
    assert result["local_gripper_actions"] == 15
    assert result["local_gripper_operations"] == 1
    assert result["ordinary_direct_actions"] == 5 and result["ordinary_direct_chunks"] == 1
    assert result["contact_direct_count_includes_local_gripper"] is True
    assert result["contact_direct_actions"] == result["local_gripper_actions"] + result["ordinary_direct_actions"]


def test_unmoved_gripper_ends_only_its_thirty_action_budget_then_requests_new_decision(tmp_path, observation):
    output = tmp_path / "trial"
    result, fake, instances = run(output, observation,
        [v2_intent(operation="close", phase="grasp"), v2_intent(direct=True, phase="recover")],
        success_at=31, move_fingers=False)
    assert result["status"] == "success", result["error"]
    assert len(fake.calls) == 2 and len(instances[0].actions) == 31
    completed = next(r for r in text_records(fake.calls[1]) if r.get("type") == "local_gripper_result")
    assert completed["actions_executed"] == 30
    assert completed["stop_reason"] == "action_budget_exhausted"
    assert completed["finger_motion_observed"] is completed["task_success_certified"] is False
    assert result["local_gripper_actions"] == 30 and result["ordinary_direct_actions"] == 1


@pytest.mark.parametrize("stop_at", [0, 7, 15])
def test_operator_stop_inside_or_after_operation_preserves_feedback_and_is_incomplete(tmp_path, observation, stop_at):
    output = tmp_path / "trial"
    result, fake, instances = run(output, observation,
        [v2_intent(operation="close", phase="grasp")], request_stop_at=stop_at)
    assert result["status"] == "user_stop", result["error"]
    assert result["steps"] == result["measured_feedback_count"] == stop_at
    assert result["pending_action_index"] is None
    assert not result["completion_time_eligible"] and not result["task_success"] and not result["libero_success"]
    assert result["success_rate"] is None and result["scored_episodes"] == 0
    assert len(fake.calls) == int(stop_at > 0) and instances[0].closed
    trace = rows(output / "trace.jsonl")
    assert trace[-1]["event"] == "operator_stop"
    assert [r["step"] for r in trace if r["event"] == "feedback"] == list(range(stop_at))


def test_bounded_quaternion_repair_is_logged_and_raw_prediction_preserved_in_runner(tmp_path, observation):
    raw = v2_intent()
    raw["field"]["orientation_target_xyzw"] = [0., 0., 0., 1.003]
    output = tmp_path / "trial"
    result, _, _ = run(output, observation, [raw], success_at=1)
    assert result["status"] == "success", result["error"]
    action = next(r for r in rows(output / "trace.jsonl") if r["event"] == "action")
    normalization = action["diagnostics"]["quaternion_normalization"]
    assert normalization["raw"] == [0., 0., 0., 1.003]
    assert normalization["normalized"] == [0., 0., 0., 1.]
    assert action["diagnostics"]["compact_prediction"] == raw


def test_video_subtraction_removes_only_control_recording_and_keeps_model_and_robot_time(
        tmp_path, observation, monkeypatch):
    clock = [100.]
    monkeypatch.setattr(runner.time, "monotonic", lambda: clock[0])

    class Recorder(FakeRecorder):
        def observe(self, *args, **kwargs):
            clock[0] += 3.  # Initial observation is outside the control loop.
            super().observe(*args, **kwargs)

        def decision(self, *args, **kwargs):
            clock[0] += 2.
            super().decision(*args, **kwargs)

        def executed(self, *args, **kwargs):
            clock[0] += 4.
            super().executed(*args, **kwargs)

        def close(self, *args, **kwargs):
            clock[0] += 7.  # Finalization must stay outside control elapsed.
            super().close(*args, **kwargs)

    result, _, _ = run(tmp_path / "trial", observation,
        [v2_intent(operation="close", phase="grasp")], success_at=7, clock=clock, recorder=Recorder)
    assert result["status"] == "success", result["error"]
    assert result["timing_components"]["video_render_s"] == pytest.approx(42.)
    assert result["timing_components"]["model_pipeline_s"] == pytest.approx(5.)
    assert result["timing_components"]["simulator_step_s"] == pytest.approx(.35)
    assert result["control_elapsed_s"] == pytest.approx(47.35)
    assert result["control_without_video_s"] == pytest.approx(5.35)
