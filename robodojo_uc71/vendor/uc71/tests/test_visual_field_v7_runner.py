"""V7 retry-to-execution accounting with an independent offline evaluator."""
import copy
from dataclasses import replace
import json

import pytest

from test_field_history import FakeTransport, envelope
from test_visual_field_protocol import observation, input_texts
from test_visual_field_v5_runner import runner, args as run_args
from test_visual_field_v6_protocol import prediction
from test_visual_field_v7_protocol import stream_error


def fixtures(clock, *, outcome="success", expected_calls=2):
    instances = []

    class Sim:
        def __init__(self, config):
            assert config.max_steps is None and config.include_arm_joints
            assert config.collision_check_mode == "task_manipulation_v1"
            self.metadata = {}; self.actions = []; self.closed = False
            instances.append(self)

        def start(self):
            pass

        def reset(self, *args):
            self.status = {"success": False, "libero_success": False,
                "collision": {"mode": "task_manipulation_v1", "detected": False,
                              "private": "V7_EVALUATOR_PRIVATE_INITIAL"}}
            return self.obs()

        def obs(self):
            obs = observation()
            gap = .04-.002*len(self.actions)
            return replace(obs, state=(*obs.state[:6], gap, -gap))

        def step(self, action):
            assert len(transport.calls) == expected_calls
            if outcome == "interrupt" and len(self.actions) == 7:
                raise KeyboardInterrupt
            clock[0] += .25
            self.actions.append(tuple(action))
            if len(self.actions) == 8:
                raw_success = outcome in ("success", "simultaneous")
                collision = outcome in ("collision", "simultaneous")
                self.status.update(success=raw_success and not collision, libero_success=raw_success,
                    collision={"mode": "task_manipulation_v1", "detected": collision,
                               "private": "V7_EVALUATOR_PRIVATE_TERMINAL"})
            return self.obs(), self.status["success"]

        def close(self):
            self.closed = True

    class Recorder:
        def __init__(self, *args):
            pass

        def frame(self, *args):
            clock[0] += .5

        def close(self):
            pass

    class Transport(FakeTransport):
        def __call__(self, endpoint, payload, headers, timeout_s):
            # Neither a truncated output nor its retry can execute an action
            # while the client's same-observation predict remains unresolved.
            assert instances and instances[0].actions == []
            assert headers["Authorization"] == "Bearer offline-oauth"
            clock[0] += 7.
            return super().__call__(endpoint, payload, headers, timeout_s)

    value = prediction(controller="gripper", phase="grasp")
    value["visual_references"] = {"external": None, "wrist": None}
    value["reason"] = "The model thinks the task is done; this statement cannot stop evaluation."
    transport = Transport(stream_error(), envelope(value, model="gpt-6-astra", reasoning={"effort":"medium"}))
    return Sim, Recorder, transport, instances


@pytest.mark.parametrize("outcome", ["success", "collision", "simultaneous", "interrupt"])
def test_missing_completion_retries_same_observation_before_action_with_exact_accounting(tmp_path, monkeypatch, outcome):
    clock = [100.]
    monkeypatch.setattr(runner.time, "monotonic", lambda: clock[0])
    Sim, Recorder, transport, sims = fixtures(clock, outcome=outcome)
    config = run_args(tmp_path, method_version="v7")
    result = runner.execute(config, token_provider=lambda: "offline-oauth", transport=transport,
                            sim_factory=Sim, recorder_factory=Recorder)
    count = 7 if outcome == "interrupt" else 8
    assert result["steps"] == result["gripper_actions"] == count
    assert result["field_actions"] == result["direct_actions"] == 0
    assert sims[0].actions == [(0.,)*6+(1.,)]*count
    assert len(transport.calls) == result["budget"]["api_calls"] == 2
    assert result["policy_diagnostic_counters"]["model_requests"] == 2
    assert result["policy_diagnostic_counters"]["intent_refreshes"] == 1
    assert result["budget"]["api_key_calls"] == 0
    assert result["budget"]["unknown_usage_calls"] == 1
    records = [next(x for x in input_texts(payload) if x["type"] == "observation")
               for payload in transport.calls]
    assert [x["observation_index"] for x in records] == [0, 0]
    for key in ("observation/state", "observation/joint_position", "observation/joint_velocity"):
        assert records[0][key] == records[1][key]
    images = [[part["image_url"] for message in payload["input"] for part in message.get("content", [])
               if part.get("type") == "input_image"] for payload in transport.calls]
    assert images[0] == images[1] and len(images[0]) == 2
    assert records[1]["observable_context"]["transport_retry"]["attempt"] == 1
    assert "V7_EVALUATOR_PRIVATE" not in json.dumps(transport.calls)
    assert all(payload["model"] == "gpt-6-astra" and payload["reasoning"] == {"effort":"medium"}
               for payload in transport.calls)
    assert result["task_success"] is (outcome == "success")
    assert result["completion_time_eligible"] is (outcome == "success")
    if outcome in ("collision", "simultaneous"):
        assert result["status"] == "collision_failure"
    if outcome == "interrupt":
        assert result["status"] == "user_stop"
    timers = result["timing_components"]
    assert timers["model_pipeline_s"] == 14.
    assert timers["failed_decision_s"] == 0.  # Retry recovered inside predict.
    assert timers["decision_s"] == 14.
    assert timers["simulator_step_s"] == .25*count
    assert timers["video_render_s"] == .5*count
    assert result["control_without_video_s"] == 14.+.25*count
    assert result["control_elapsed_s"] == result["control_without_video_s"]+timers["video_render_s"]
    retry_records = list((config.output/"api").glob("*.v7_retry.json"))
    assert len(retry_records) == 1
    assert json.loads(retry_records[0].read_text())["actions_executed_from_failed_response"] == 0
    manifest = json.loads((config.output/"manifest.json").read_text())
    assert manifest["method_version"] == "v7" and manifest["shots"] == 0
    assert manifest["max_policy_steps"] is None and "transport_retry" in manifest
    assert sims[0].closed and result["cleanup_errors"] == []


@pytest.mark.parametrize("failure,expected_calls", [("repeat_stream", 2), ("http_auth", 1), ("other_api", 1)])
def test_terminal_transport_failure_never_executes_and_charges_entire_failed_predict(tmp_path, monkeypatch, failure, expected_calls):
    clock = [100.]
    monkeypatch.setattr(runner.time, "monotonic", lambda: clock[0])
    Sim, Recorder, transport, sims = fixtures(clock, expected_calls=expected_calls)
    if failure == "repeat_stream":
        transport.responses = [stream_error(), stream_error()]
    elif failure == "http_auth":
        transport.responses = [stream_error(status=401)]
    else:
        transport.responses = [stream_error(code="connection_failed")]
    config = run_args(tmp_path, method_version="v7")
    result = runner.execute(config, token_provider=lambda: "offline-oauth", transport=transport,
                            sim_factory=Sim, recorder_factory=Recorder)
    assert result["status"] == "subscription_error" and result["error"]["type"] == "APIError"
    assert result["steps"] == 0 and sims[0].actions == []
    assert len(transport.calls) == result["budget"]["api_calls"] == expected_calls
    assert result["policy_diagnostic_counters"]["model_requests"] == expected_calls
    assert result["policy_diagnostic_counters"]["intent_refreshes"] == 0
    assert result["budget"]["api_key_calls"] == 0
    assert result["timing_components"]["failed_decision_s"] == 7.*expected_calls
    assert result["timing_components"]["decision_s"] == 7.*expected_calls
    assert result["timing_components"]["model_pipeline_s"] == 0.
    assert result["control_without_video_s"] == 7.*expected_calls
    assert not result["task_success"] and not result["completion_time_eligible"]
    assert sims[0].closed and result["cleanup_errors"] == []


@pytest.mark.parametrize("changes", [{"controller":"direct"}, {"demo":"not-a-demo.json"}])
def test_v7_unrequested_baseline_or_demo_fails_before_auth_or_output_creation(tmp_path, changes):
    with pytest.raises(ValueError, match="zero-shot Hybrid"):
        runner.execute(run_args(tmp_path, method_version="v7", **changes),
                       token_provider=lambda: pytest.fail("Must reject before auth"))
    assert not (tmp_path/"trial").exists()
