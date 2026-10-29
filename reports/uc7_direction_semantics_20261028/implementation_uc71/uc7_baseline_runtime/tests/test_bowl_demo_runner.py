"""Matched bowl demonstration runner selection and measured-action execution."""
import json

import pytest

from gpt_field.demonstration import load_demonstration
from gpt_field.distance_tasks import resolve_distance_task
from test_demonstration import make_bundle
from test_distance_policy import intent
from test_distance_runner import SETTINGS as HYBRID_OPTIONS
from test_field_history import FakeTransport, envelope
from test_field_history_runner import FakeRecorder, fake_simulator, observation, rows
from test_hybrid_subscription_runner import runner, sim_config


BOWL = resolve_distance_task("bowl_plate")
DIRECT_OPTIONS = dict(controller_mode="direct", reasoning_effort="medium", execution_mode="fixed",
    feedback_history="block_summary", centered_placement_guidance=True, grasp_settle_guidance=True)


def bowl_reference(tmp_path):
    bundle = make_bundle()
    bundle.update(task_key=BOWL.key, task_prompt=BOWL.prompt)
    bundle["source"].update(suite=BOWL.suite, task_id=BOWL.task_id,
                            file_name=BOWL.prompt.replace(" ", "_") + "_demo.hdf5")
    path = tmp_path / "bowl_demo.json"
    path.write_text(json.dumps(bundle))
    return path, load_demonstration(path, task_key=BOWL.key)


@pytest.mark.parametrize("direct", [False, True])
def test_bowl_demo_cli_task_defaults_and_reference_identity(tmp_path, capsys, direct):
    path, reference = bowl_reference(tmp_path)
    mode = ["--controller-mode", "direct"] if direct else ["--distance-field-control"]
    assert runner.main(["--task", BOWL.key, "--demonstration", str(path), "--preflight", *mode]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["distance_task"] == BOWL.key and result["task_id"] == result["episode_index"] == 0
    assert result["simulator"]["suite"] == BOWL.suite and result["simulator"]["seed"] == 7
    assert result["demonstration"] == {"artifact": "demonstration.json", **reference.metadata()}
    assert result["policy"]["reasoning_effort"] == "medium"
    assert result["policy"]["controller_mode"] == ("direct" if direct else "hybrid")
    assert result["policy"]["execution_mode"] == ("fixed" if direct else "tracking")
    assert result["policy"]["feedback_history"] == "block_summary"
    assert result["policy"]["distance_field_control"] is not direct
    assert result["api_calls"] == result["api_key_calls"] == 0
    assert not result["stop_on_model_complete"]
    assert result["max_policy_steps"] is result["max_wall_time_s"] is None


def test_bowl_direct_without_demo_keeps_historical_client(capsys):
    assert runner.main(["--task", BOWL.key, "--controller-mode", "direct", "--preflight"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert "demonstration" not in result and "demo_direct_control" not in result["policy"]
    assert result["policy"]["reasoning_effort"] == "high"
    assert result["policy"]["feedback_history"] == "full"


@pytest.mark.parametrize("direct", [False, True])
def test_bowl_pair_keeps_demo_separate_and_executes_current_model_actions(tmp_path, observation, direct):
    path, reference = bowl_reference(tmp_path)
    output = tmp_path / "run"
    base, instances = fake_simulator(observation, success_at=7)

    class BowlSimulator(base):
        def reset(self, task_id, episode_index):
            assert (task_id, episode_index) == (0, 0)
            assert (output / "demonstration.json").read_bytes() == path.read_bytes()
            return super().reset(task_id, episode_index)

    contact = intent(direct=True, phase="grasp")
    contact["direct"]["actions"] = [[.013, 0., 0., 0., 0., 0., 1.]] * 5
    raw = ({"actions": contact["direct"]["actions"], "target_region": "black_bowl",
            "routing_reason": "Fixture grasp from current observation.",
            "control_context": contact["control_context"]} if direct else contact)
    decisions = [raw, raw] if direct else [intent(), raw, raw]
    transport = FakeTransport(*(envelope(response, id=f"bowl-demo-{index}", model="gpt-6-astra",
        reasoning={"effort": "medium"}) for index, response in enumerate(decisions)))
    result = runner.execute_episode(output, config=runner.configuration(), sim_config=sim_config(),
        task_key=BOWL.key, demonstration_path=path, token_provider=lambda: "fixture-oauth",
        transport=transport, sim_factory=BowlSimulator, recorder_factory=FakeRecorder,
        **(DIRECT_OPTIONS if direct else HYBRID_OPTIONS))
    assert result["status"] == "success", result["error"]
    assert result["steps"] == 7 and result["libero_success"] is True
    assert result["distance_task"] == BOWL.key and result["api_key_calls"] == 0
    assert result["demonstration"] == {"artifact": "demonstration.json", **reference.metadata()}
    assert result["coarse_field_actions"] == (0 if direct else 1)
    assert result["local_plan_count"] == (0 if direct else 1)
    assert instances[0].actions[(0 if direct else 1):] == [contact["direct"]["actions"][0]] * (7 if direct else 6)
    assert all(payload["input"][0] == reference.message() for payload in transport.calls)
    assert all(sum(msg == reference.message() for msg in payload["input"]) == 1 for payload in transport.calls)
    assert "MUST_NOT_ENTER_POLICY" not in json.dumps(transport.calls)
    assert [r["step"] for r in rows(output / "trace.jsonl") if r["event"] == "executed" and r["success"]] == [6]


@pytest.mark.parametrize("direct", [False, True])
def test_wrong_task_demo_is_rejected_before_auth_or_output(tmp_path, direct):
    path = tmp_path / "drawer_demo.json"
    path.write_text(json.dumps(make_bundle()))
    output = tmp_path / "must-not-exist"
    with pytest.raises(ValueError, match="task"):
        runner.execute_episode(output, config=runner.configuration(), sim_config=sim_config(),
            task_key=BOWL.key, demonstration_path=path,
            token_provider=lambda: pytest.fail("Mismatched demo requested credentials"),
            **(DIRECT_OPTIONS if direct else HYBRID_OPTIONS))
    assert not output.exists()
