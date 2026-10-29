"""A fixed external demonstration must not become live execution history."""
from dataclasses import replace
import json

import pytest

from gpt_field.demonstration import load_demonstration
from gpt_field.distance_subscription import ChatGPTPlanDistanceClient, distance_prompt_version
from test_demonstration import make_bundle
from test_distance_runner import SETTINGS
from test_drawer_subscription import drawer_observation, drawer_prediction
from test_field_history import FakeTransport, envelope
from test_field_history_runner import FakeRecorder, fake_simulator, observation, rows
from test_hybrid_subscription_runner import runner, sim_config


def reference(tmp_path):
    path = tmp_path / "reference.json"
    path.write_text(json.dumps(make_bundle()))
    return path, load_demonstration(path)


def test_demo_is_once_per_payload_and_never_committed_or_trimmed_as_live_history(tmp_path):
    _, demo = reference(tmp_path)
    prediction = drawer_prediction(contact=True, phase="align")
    transport = FakeTransport(*(envelope(prediction, id=f"demo-condition-{i}",
        model="gpt-6-astra", reasoning={"effort": "medium"}) for i in range(4)))
    client = ChatGPTPlanDistanceClient(task_key="drawer_bowl", demonstration=demo,
        token_provider=lambda: "fixture-oauth", transport=transport)
    for i in range(4):
        client.predict(drawer_observation(i).to_dict())
    for payload in transport.calls:
        assert payload["input"][0] == demo.message()
        assert sum(message == demo.message() for message in payload["input"]) == 1
        assert payload["model"] == "gpt-6-astra" and payload["reasoning"] == {"effort": "medium"}
        assert "AFTER its labeled action_index" in payload["instructions"]
        assert "demonstrations or future images are supplied" not in payload["instructions"]
        assert "No reference action is executed automatically" in payload["instructions"]
        live_images = sum(part.get("type") == "input_image"
            for message in payload["input"][1:] for part in message.get("content", [])
            if isinstance(part, dict))
        assert live_images <= 4
    assert demo.message() not in client.history_snapshot()
    assert client.history_stats()["observations"] == 4
    assert client.history_stats()["feedback_records"] == 0
    assert client.history_stats()["retained_image_pairs"] == 2
    assert client.prompt_version == distance_prompt_version("drawer_bowl", demonstration=True)
    client.reset_history()
    assert client.history_stats()["observations"] == 0
    assert client.build_payload(drawer_observation().to_dict())["input"][0] == demo.message()


def test_demo_cli_preview_preserves_reference_identity_and_current_protocol(tmp_path, capsys):
    path, demo = reference(tmp_path)
    assert runner.main(["--distance-field-control", "--task", "drawer_bowl",
                        "--demonstration", str(path), "--preflight"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["demonstration"] == {"artifact": "demonstration.json", **demo.metadata()}
    assert result["api_calls"] == 0 and not result["credentials_checked"]
    assert result["max_policy_steps"] is result["max_wall_time_s"] is None
    assert result["policy"]["direct_action_steps"] == 5
    assert result["policy"]["prompt_version"] == distance_prompt_version("drawer_bowl", demonstration=True)
    with pytest.raises(SystemExit):
        runner.main(["--demonstration", str(path), "--preflight"])


def test_demo_is_archived_before_mock_execution_and_actions_come_from_live_decisions(tmp_path, observation):
    path, demo = reference(tmp_path)
    obs = replace(observation, prompt=demo.metadata()["task_prompt"])
    base, instances = fake_simulator(obs, success_at=6)

    class DrawerSimulator(base):
        def list_tasks(self):
            return [{"task_id": 3, "prompt": obs.prompt}]

        def reset(self, task_id, episode_index):
            assert (task_id, episode_index) == (3, 0)
            assert (tmp_path / "run" / "demonstration.json").is_file()
            return super().reset(0, 0)

    decision = drawer_prediction(contact=True, phase="align")
    decision["direct"]["actions"] = [[0., .012, 0., 0., 0., 0., -1.]] * 5
    transport = FakeTransport(*(envelope(decision, id=f"demo-run-{i}", model="gpt-6-astra",
        reasoning={"effort": "medium"}) for i in range(2)))
    output = tmp_path / "run"
    result = runner.execute_episode(output, config=runner.configuration(),
        sim_config=sim_config(suite="libero_goal"), task_key="drawer_bowl", demonstration_path=path,
        token_provider=lambda: "fixture-oauth", transport=transport, sim_factory=DrawerSimulator,
        recorder_factory=FakeRecorder, **SETTINGS)
    assert result["status"] == "success" and result["steps"] == 6
    assert result["api_calls"] == 2 and result["api_key_calls"] == 0
    assert result["demonstration"] == {"artifact": "demonstration.json", **demo.metadata()}
    assert (output / "demonstration.json").read_bytes() == path.read_bytes()
    manifest = json.loads((output / "manifest.json").read_text())["configuration"]
    assert manifest["demonstration"] == result["demonstration"]
    assert all(action == decision["direct"]["actions"][0] for action in instances[0].actions)
    assert sum(r["event"] == "executed" for r in rows(output / "trace.jsonl")) == 6
    assert all(payload["input"][0] == demo.message() for payload in transport.calls)
