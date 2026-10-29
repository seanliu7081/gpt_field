"""New runner preset stays opt-in, subscription-only and LIBERO-stopped."""
import json

import pytest

from test_distance_policy import intent
from test_field_history import FakeTransport, envelope
from test_field_history_runner import FakeRecorder, fake_simulator, observation, rows
from test_hybrid_subscription_runner import runner, sim_config


SETTINGS = dict(distance_field_control=True, field_improvements=True,
                reasoning_effort="medium", execution_mode="tracking", feedback_history="block_summary",
                centered_placement_guidance=True, grasp_settle_guidance=True)


def test_cli_preset_is_dry_and_has_no_episode_caps(capsys):
    assert runner.main(["--distance-field-control", "--preflight"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["api_calls"] == result["api_key_calls"] == 0
    assert not result["credentials_checked"] and not result["simulator_started"]
    assert result["payload"]["reasoning"] == {"effort": "medium"}
    assert result["policy"]["distance_field_control"]
    assert result["policy"]["local_path_strategy"] == "dijkstra_distance_field"
    assert result["policy"]["field_reuse_limit"] == 120
    assert result["policy"]["direct_action_steps"] == 5
    assert result["max_policy_steps"] is result["max_wall_time_s"] is None
    assert not result["stop_on_model_complete"]


@pytest.mark.parametrize("flag", [
    ["--reasoning-effort", "high"], ["--execution-mode", "fixed"],
    ["--feedback-history", "full"], ["--field-representation", "dense"],
    ["--controller-mode", "direct"], ["--stop-on-model-complete"], ["--offline-smoke"],
])
def test_cli_rejects_incompatible_protocol_changes(flag):
    with pytest.raises(SystemExit) as exc:
        runner.main(["--distance-field-control", "--preflight", *flag])
    assert exc.value.code == 2


def test_runner_records_new_policy_local_feedback_and_unchanged_direct(tmp_path, observation):
    sim, instances = fake_simulator(observation, success_at=6)
    direct = intent(direct=True, phase="release")
    fake = FakeTransport(*(envelope(direct, id=f"fixture-{i}", reasoning={"effort": "medium"})
                           for i in range(2)))
    output = tmp_path / "fixture"
    result = runner.execute_episode(output, config=runner.configuration(), sim_config=sim_config(),
        token_provider=lambda: "fixture-oauth", transport=fake, sim_factory=sim,
        recorder_factory=FakeRecorder, **SETTINGS)
    assert result["status"] == "success" and result["steps"] == 6
    assert result["distance_field_control"] and result["api_key_calls"] == 0
    assert result["timing_components"]["local_visual_s"] >= 0
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["configuration"]["policy"]["class"] == "gpt_field.distance_policy.DistanceHybridPolicy"
    records = rows(output / "trace.jsonl")
    assert sum(r["event"] == "local_visual_feedback" for r in records) == 6
    events = [r for r in records if r["event"] == "distance_control_event"]
    assert len(events) == 1 and events[0]["step"] == 5
    assert events[0]["distance_control_event"]["reason"] == "placement_unverified"
    assert all(tuple(a) == tuple(direct["direct"]["actions"][0]) for a in instances[0].actions)
