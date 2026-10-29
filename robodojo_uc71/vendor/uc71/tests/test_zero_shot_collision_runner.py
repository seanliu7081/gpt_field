"""Zero-shot paired execution stops on forbidden contact before any next action."""
import hashlib
import json

import pytest

from test_bowl_demo_runner import DIRECT_OPTIONS, HYBRID_OPTIONS
from test_demo_direct_subscription import prediction as direct_prediction
from test_distance_policy import intent
from test_field_history import FakeTransport, envelope
from test_field_history_runner import FakeRecorder, fake_simulator, observation, rows
from test_hybrid_subscription_runner import runner, sim_config
from test_subscription_scene_runner import scene


@pytest.mark.parametrize("direct", [False, True])
def test_cli_zero_shot_preflight_has_no_demo_and_identical_collision_contract(tmp_path, capsys, direct):
    _, custom, state = scene(tmp_path)
    mode = ["--controller-mode", "direct"] if direct else ["--distance-field-control"]
    assert runner.main(["--zero-shot-collision-free", "--preflight", "--task-bddl-override", str(custom),
        "--initial-state-override", str(state), *mode]) == 0
    p = json.loads(capsys.readouterr().out)
    assert p["zero_shot_collision_free"] and "demonstration" not in p
    assert p["policy"]["reasoning_effort"] == "medium" and p["policy"]["direct_action_steps"] == 5
    assert p["simulator"]["collision_check_mode"] == "bowl_obstacles_v1"
    assert p["policy"]["method"] == ("zero_shot_direct" if direct else "hybrid_field_direct")
    assert not p["files_written"] and not p["credentials_checked"] and not p["simulator_started"]
    assert "zero-shot" in p["payload"]["instructions"].lower()
    assert any("collision" in reason for reason in p["stopping_rule"])


@pytest.mark.parametrize("direct", [False, True])
@pytest.mark.parametrize("outcome", ["success", "collision", "simultaneous", "initial_collision", "disabled"])
def test_zero_shot_collision_scoring_stops_and_keeps_privileged_contact_private(tmp_path, observation, direct, outcome):
    original, custom, state = scene(tmp_path)
    output = tmp_path / "run"
    base, instances = fake_simulator(observation, success_at=1)
    custom_hash = hashlib.sha256(custom.read_bytes()).hexdigest()

    def collision(detected, initial=False):
        return {"enabled": True, "mode": "bowl_obstacles_v1", "detected": detected,
                "first_event": {"action_index": None if initial else 0, "physics_substep": 3,
                    "private_geometry": "CONTACT_MUST_NOT_ENTER_POLICY"} if detected else None}

    class CollisionSimulator(base):
        def reset(self, task_id, episode_index):
            obs = super().reset(task_id, episode_index)
            assert not (output / "demonstration.json").exists()
            self.status.update(original_bddl_file=str(original),
                original_bddl_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),
                bddl_sha256=custom_hash, task_bddl_override_sha256=custom_hash,
                libero_success=False, collision=collision(outcome == "initial_collision", True))
            self.status["terminated"] = outcome == "initial_collision"
            if outcome == "disabled":
                del self.status["collision"]
            return obs

        def step(self, action):
            obs, _ = super().step(action)
            detected = outcome in ("collision", "simultaneous")
            raw = outcome in ("success", "simultaneous")
            self.status.update(libero_success=raw, success=raw and not detected,
                terminated=raw or detected, collision=collision(detected))
            return obs, self.status["success"]

        def close(self):
            super().close()
            # Real SimProcess.close clears this protocol snapshot. The runner
            # must retain the raw goal result before cleanup, including on collision.
            self.status = {}

    raw = direct_prediction(phase="grasp", target="black_bowl") if direct else intent(direct=True, phase="grasp")
    transport = FakeTransport(envelope(raw, model="gpt-6-astra", reasoning={"effort": "medium"}))
    result = runner.execute_episode(output, config=runner.configuration(),
        sim_config=sim_config(task_bddl_override=str(custom), initial_state_override=str(state),
            collision_check_mode="bowl_obstacles_v1"), zero_shot_collision_free=True,
        token_provider=lambda: "offline-oauth", transport=transport,
        sim_factory=CollisionSimulator, recorder_factory=FakeRecorder,
        **(DIRECT_OPTIONS if direct else HYBRID_OPTIONS))
    assert instances[0].closed
    if outcome in ("initial_collision", "disabled"):
        assert result["status"] == "infrastructure_error"
        assert not result["evaluation_completed"] and result["scored_episodes"] == 0
        assert not transport.calls and not instances[0].actions
        return
    assert len(transport.calls) == len(instances[0].actions) == result["steps"] == 1
    assert result["status"] == ("success" if outcome == "success" else "collision_failure"), result["error"]
    assert result["task_success"] == (outcome == "success")
    assert result["libero_success"] == (outcome != "collision")
    assert result["evaluation_completed"] and result["benchmark_completed"]
    assert result["success_rate"] == (1.0 if outcome == "success" else 0.0)
    assert result["scored_episodes"] == 1 and result["incomplete_episodes"] == 0
    assert json.loads((output / "task_evaluation.json").read_text())["task_success"] == (outcome == "success")
    assert json.loads((output / "collision.json").read_text())["detected"] == (outcome != "success")
    events = rows(output / "trace.jsonl")
    assert sum(r["event"] == "forbidden_collision" for r in events) == (outcome != "success")
    payload = json.dumps(transport.calls)
    for private in ("external_demonstration", "CONTACT_MUST_NOT_ENTER_POLICY", "MUST_NOT_ENTER_POLICY", custom_hash):
        assert private not in payload
    assert not (output / "demonstration.json").exists()


@pytest.mark.parametrize("extra", [["--demonstration", "demo.json"], ["--stop-on-model-complete"],
                                   ["--timing-comparison"], ["--reasoning-effort", "high"]])
def test_unsupported_zero_shot_options_fail_before_auth_or_configuration(tmp_path, monkeypatch, extra):
    monkeypatch.setattr(runner, "configuration", lambda: pytest.fail("Invalid options reached configuration"))
    with pytest.raises(SystemExit) as stopped:
        runner.main(["--zero-shot-collision-free", "--distance-field-control", "--task-bddl-override", "task.bddl",
            "--initial-state-override", "state.npy", "--output", str(tmp_path / "absent"), *extra])
    assert stopped.value.code == 2 and not (tmp_path / "absent").exists()


def test_zero_shot_does_not_silently_accept_missing_collision_monitor(tmp_path):
    _, custom, state = scene(tmp_path)
    with pytest.raises(ValueError, match="monitoring"):
        runner.preflight(runner.configuration(), sim_config(task_bddl_override=str(custom),
            initial_state_override=str(state)), zero_shot_collision_free=True, **HYBRID_OPTIONS)
