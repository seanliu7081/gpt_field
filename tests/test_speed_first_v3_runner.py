"""Native-RGB v3 protocol boundaries; no model or simulator is contacted."""
from dataclasses import replace
import hashlib
import json

import numpy as np
import pytest

from gpt_field.observation import PolicyObservation, decode_image, image_to_data_url
from gpt_field.speed_first_v3 import ChatGPTPlanSpeedFirstV3Client, SPEED_FIRST_V3_VERSION
from test_distance_runner import SETTINGS
from test_field_history import FakeTransport, envelope
from test_field_history_runner import FakeRecorder, observation, rows
from test_hybrid_subscription_runner import runner, sim_config
from test_speed_first_runner import monitored_simulator
from test_speed_first_v2_runner import v2_intent


OPTIONS = {**SETTINGS, "speed_first": True, "speed_first_v2": True, "speed_first_v3": True}


@pytest.fixture
def highres_observation(observation):
    # This image is generated at512, not enlarged from the historical224 input.
    yy, xx = np.indices((512, 512))
    rgb = np.stack((xx % 256, yy % 256, (xx * 17 + yy * 31) % 256), axis=-1).astype(np.uint8)
    return replace(observation, image=image_to_data_url(rgb, size=512, rotate=False),
                   wrist_image=image_to_data_url(rgb[:, ::-1], size=512, rotate=False))


def v3_intent(**kwargs):
    return v2_intent(**kwargs)


def run(output, obs, responses, *, success_at=None, collision_at=1,
        request_stop_at=None, clock=None, recorder=FakeRecorder):
    base, instances = monitored_simulator(obs, success_at=success_at,
        collision_at=collision_at, output=output, request_stop_at=request_stop_at)

    class Simulator(base):
        def step(self, action):
            measured, success = super().step(action)
            if action[6] != 0:
                state = list(measured.state)
                target = .016 if action[6] > 0 else .04
                magnitude = abs(state[6])
                magnitude += max(-.004, min(.004, target - magnitude))
                state[6:8] = [magnitude, -magnitude]
                self.obs = replace(measured, state=tuple(state))
            if clock is not None:
                clock[0] += .05
            return self.obs, success

    fake = FakeTransport(*(envelope(raw, id=f"v3-fixture-{index}", model="gpt-6-astra",
        reasoning={"effort": "medium"}) for index, raw in enumerate(responses)))

    def transport(*args):
        if clock is not None:
            clock[0] += 5.
        return fake(*args)

    result = runner.execute_episode(output, config=runner.configuration(),
        sim_config=sim_config(render_size=512, image_size=512,
            collision_check_mode="task_manipulation_v1", stop_on_collision=False),
        token_provider=lambda: "offline-v3-oauth", transport=transport,
        sim_factory=Simulator, recorder_factory=recorder, **OPTIONS)
    return result, fake, instances


def test_v3_preflight_has_native_512_lazy_1mm_and_no_side_effects(tmp_path, monkeypatch, capsys):
    observed = []
    build = ChatGPTPlanSpeedFirstV3Client.build_payload

    def capture(self, obs):
        observed.append(obs)
        return build(self, obs)

    monkeypatch.setattr(ChatGPTPlanSpeedFirstV3Client, "build_payload", capture)
    monkeypatch.setattr(runner, "SimProcess", lambda *a: pytest.fail("preflight started simulator"))
    monkeypatch.setattr(ChatGPTPlanSpeedFirstV3Client, "predict", lambda *a, **k: pytest.fail("model called"))
    output = tmp_path / "absent"
    assert runner.main(["--speed-first-v3", "--preflight", "--output", str(output)]) == 0
    record = json.loads(capsys.readouterr().out)
    assert not output.exists()
    assert record["speed_first"] and record["speed_first_v2"] and record["speed_first_v3"]
    assert record["speed_first_protocol"] == record["policy"]["prompt_version"] == SPEED_FIRST_V3_VERSION
    assert record["payload"]["model"] == "gpt-6-astra"
    assert record["payload"]["reasoning"] == {"effort": "medium"}
    assert record["simulator"]["render_size"] == record["simulator"]["image_size"] == 512
    assert record["map_spacing_m"] == .001 and record["map_size"] == 161
    assert record["map_node_count"] == 4173281 and record["map_coverage_m"] == .16
    assert record["policy"]["grid"]["size"] == 161
    assert record["policy"]["grid"]["spacing_m"] == .001
    assert record["policy"]["local_path_strategy"] == "analytic_clear_field_else_lazy_26_neighbor_astar_1mm"
    assert len(observed) == 1
    for key in ("observation/image", "observation/wrist_image"):
        assert decode_image(observed[0][key]).size == (512, 512)
    assert set(observed[0]) == {"observation/image", "observation/wrist_image", "observation/state", "prompt"}
    assert record["billing"] == "chatgpt_subscription" and record["api_key_fallback"] is False
    assert record["simulator"]["seed"] == 7 and record["task_id"] == record["episode_index"] == 0
    assert record["simulator"]["stop_on_collision"] is False
    assert record["simulator"]["stop_on_success"] is True
    assert record["success_qualification"] == "raw_libero_goal" and not record["stop_on_model_complete"]
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
def test_v3_cli_rejects_protocol_changes_before_side_effects(flags, tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "configuration", lambda: pytest.fail("invalid arguments reached configuration"))
    output = tmp_path / "absent"
    with pytest.raises(SystemExit) as caught:
        runner.main(["--speed-first-v3", "--output", str(output), *flags])
    assert caught.value.code == 2 and not output.exists()


def test_high_resolution_requires_v3_and_v3_rejects_upsampled_render_configuration():
    spec = runner.resolve_distance_task("bowl_plate")
    config = runner.configuration()
    with pytest.raises(ValueError, match="512"):
        runner._validate_task_configuration(config, sim_config(render_size=256, image_size=512),
                                            spec, speed_first_v3=True)
    with pytest.raises(ValueError, match="512"):
        runner._validate_task_configuration(config, sim_config(), spec, speed_first_v3=True)
    with pytest.raises(ValueError, match="256/224"):
        runner._validate_task_configuration(config, sim_config(render_size=512, image_size=512), spec)


def test_actual_v3_wire_uses_only_oauth_and_strict_compact_schema(highres_observation, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "OFFLINE_POISON_API_KEY_MUST_NOT_BE_USED")
    captured = []

    def transport(endpoint, payload, headers, timeout):
        captured.append((endpoint, payload, headers, timeout))
        return envelope(v3_intent(direct=True), model="gpt-6-astra", reasoning={"effort": "medium"})

    client = ChatGPTPlanSpeedFirstV3Client(token_provider=lambda: "OFFLINE_SUBSCRIPTION_OAUTH",
                                         transport=transport)
    result = client.predict(highres_observation.to_dict())
    assert result["controller"] == "contact_direct" and len(result["direct"]["actions"]) == 5
    assert client._api_key == ""
    endpoint, payload, headers, timeout = captured[0]
    assert endpoint == client.config.endpoint
    assert headers["Authorization"] == "Bearer OFFLINE_SUBSCRIPTION_OAUTH"
    assert "OFFLINE_POISON_API_KEY" not in json.dumps(captured)
    assert timeout == 180
    assert payload["store"] is False and payload["stream"] is True
    assert payload["include"] == ["reasoning.encrypted_content"]
    assert payload["reasoning"] == {"effort": "medium"}
    assert "max_output_tokens" not in payload and "previous_response_id" not in payload
    schema = payload["text"]["format"]
    assert schema["strict"] is True and schema["name"] == "astra_speed_first_distance_hybrid_v3"

    def strict_objects(node):
        if isinstance(node, dict):
            if node.get("type") == "object":
                assert node.get("additionalProperties") is False
                assert set(node.get("required", [])) == set(node["properties"])
                assert "values" not in node["properties"]
            for value in node.values():
                strict_objects(value)
        elif isinstance(node, list):
            for value in node:
                strict_objects(value)

    strict_objects(schema["schema"])
    image_parts = [part for msg in payload["input"] for part in msg.get("content", [])
                   if part.get("type") == "input_image"]
    assert len(image_parts) == 2
    assert all(part["detail"] == "high" and decode_image(part["image_url"]).size == (512, 512)
               for part in image_parts)
    assert client.budget_snapshot()["subscription_inference_calls"] == 1


def test_native512_observation_preserves_detail_and_still_excludes_oracles():
    rng = np.random.default_rng(11)
    pixels = rng.integers(0, 256, (512, 512, 3), dtype=np.uint8)
    raw = {"agentview_image": pixels, "robot0_eye_in_hand_image": pixels[:, ::-1],
           "robot0_eef_pos": [0., 0., 1.], "robot0_eef_quat": [0., 0., 0., 1.],
           "robot0_gripper_qpos": [.04, -.04], "object_pose": "MUST_NOT_ENTER_POLICY",
           "depth": "MUST_NOT_ENTER_POLICY", "camera_calibration": "MUST_NOT_ENTER_POLICY"}
    measured = PolicyObservation.from_libero(raw, "fixture", image_size=512)
    assert np.array_equal(np.asarray(decode_image(measured.image)), pixels[::-1, ::-1])
    assert "MUST_NOT_ENTER_POLICY" not in json.dumps(measured.to_dict())
    assert len(measured.state) == 8


def test_local_close_then_direct5_then_lazy_field_have_native_inputs_and_truthful_summary(
        tmp_path, highres_observation):
    direct = v3_intent(direct=True, phase="verify")
    direct["direct"]["actions"] = [[.01 * (i + 1), 0., 0., 0., 0., 0., 1.] for i in range(5)]
    output = tmp_path / "trial"
    result, fake, instances = run(output, highres_observation, [
        v3_intent(operation="close", phase="grasp"), direct,
        v3_intent(phase="recover", goal=(0., -.03, 0.))], success_at=21)
    assert result["status"] == "success", result["error"]
    assert result["libero_success"] is result["task_success"] is True
    assert result["collision"]["detected"] and result["failure_reason"] is None
    assert result["steps"] == result["measured_feedback_count"] == 21
    assert result["local_gripper_actions"] == 15 and result["ordinary_direct_actions"] == 5
    assert result["coarse_field_actions"] == 1 and len(fake.calls) == 3
    assert instances[0].actions[15:20] == direct["direct"]["actions"]
    assert instances[0].closed
    assert "CONTACT_MUST_NOT_ENTER_POLICY" not in json.dumps(fake.calls)
    assert "MUST_NOT_ENTER_POLICY" not in json.dumps(fake.calls)
    for request in fake.calls:
        images = [part for msg in request["input"] for part in msg.get("content", [])
                  if part.get("type") == "input_image"]
        assert images and all(decode_image(part["image_url"]).size == (512, 512) for part in images)
    manifest = json.loads((output / "manifest.json").read_text())["configuration"]
    assert manifest["policy"]["class"] == "gpt_field.speed_first_v3.SpeedFirstV3Policy"
    assert manifest["simulator"]["render_size"] == manifest["simulator"]["image_size"] == 512
    for name in ("speed_first_v3.py", "speed_first_v3_video.py"):
        assert manifest["software"]["source_sha256"][name] == hashlib.sha256((output / "frozen_source" / name).read_bytes()).hexdigest()
    records = rows(output / "trace.jsonl")
    assert [r["step"] for r in records if r["event"] == "feedback"] == list(range(21))
    assert result["control_without_video_s"] == pytest.approx(
        result["control_elapsed_s"] - result["timing_components"]["video_render_s"])


def test_libero_stops_inside_local_operation_without_waiting_for_model(tmp_path, highres_observation):
    result, fake, instances = run(tmp_path / "early", highres_observation,
        [v3_intent(operation="close", phase="grasp")], success_at=3)
    assert result["status"] == "success", result["error"]
    assert result["steps"] == result["local_gripper_actions"] == 3
    assert result["model_declared_complete"] is False
    assert len(fake.calls) == 1 and instances[0].closed


def test_operator_stop_preserves_complete_feedback_and_incomplete_status(tmp_path, highres_observation):
    output = tmp_path / "stop"
    result, fake, instances = run(output, highres_observation,
        [v3_intent(operation="close", phase="grasp")], request_stop_at=7)
    assert result["status"] == "user_stop", result["error"]
    assert result["steps"] == result["measured_feedback_count"] == 7
    assert result["pending_action_index"] is None and not result["completion_time_eligible"]
    assert result["success_rate"] is None and result["scored_episodes"] == 0
    assert len(fake.calls) == 1 and instances[0].closed
