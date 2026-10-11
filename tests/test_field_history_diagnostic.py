"""Terminal/provenance guards only; these tests never instantiate LIBERO or Astra."""

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/diagnose_field_history_replay.py"
spec = importlib.util.spec_from_file_location("diagnose_field_history_replay", SCRIPT)
diagnostic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostic)


def write(path, value):
    path.write_text(json.dumps(value))


@pytest.fixture
def terminal_source(tmp_path):
    source = tmp_path / "field_history_terminal"
    source.mkdir()
    state = [0, 0, 1, 0, 0, 0, .04, -.04]
    after = [0, 0, 1, 0, 0, 0, .03, -.03]
    action = [0, 0, 0, 0, 0, 0, 1]

    def observation(values):
        return {"prompt": diagnostic.TASK, "observation/state": values,
                "observation/image": "fixture_not_rendered", "observation/wrist_image": "fixture_not_rendered"}

    reset_status = {"task_id": 0, "episode_index": 0, "steps": 0, "success": False}
    executed_status = {**reset_status, "steps": 1}
    configuration = {
        "kind": "fresh_field_history_experiment", "mode": "live_astra_field_history",
        "task": diagnostic.TASK, "task_id": 0, "episode_index": 0,
        "simulator": {"suite": "libero_spatial", "seed": 7, "max_steps": None,
                      "settle_steps": 10, "image_size": 224, "render_size": 256},
        "software": {"source_sha256": {"schema.py": diagnostic.sha(ROOT / "src/gpt_field/schema.py")}},
        "runner": {"path": "scripts/evaluate_bowl_field_history.py",
                   "sha256": diagnostic.sha(ROOT / "scripts/evaluate_bowl_field_history.py")},
    }
    write(source / "manifest.json", {"configuration": configuration,
                                     "identity_hash": diagnostic.digest(configuration)})
    write(source / "run_state.json", {"status": "budget_exceeded", "stage": "finished", "step": 1})
    write(source / "summary.json", {"status": "budget_exceeded", "steps": 1, "measured_feedback_count": 1})
    write(source / "simulator.json", {"metadata": {}, "tasks": []})
    rows = [
        {"event": "reset", "observation": observation(state), "sim_status": reset_status},
        {"event": "action", "step": 0, "state": state, "action": action},
        {"event": "executed", "step": 0, "action": action, "next_state": after,
         "observation": observation(after), "sim_status": executed_status, "success": False},
        {"event": "feedback", "step": 0, "feedback": {
            "before_state": state, "after_state": after, "normalized_action": action}},
    ]
    (source / "trace.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    return source


def rewrite_identity(source, **changes):
    manifest = diagnostic.read(source / "manifest.json")
    manifest["configuration"].update(changes)
    manifest["identity_hash"] = diagnostic.digest(manifest["configuration"])
    write(source / "manifest.json", manifest)


def test_replay_implementation_changes_only_experiment_identity():
    # All reconstruction, exact-state/pixel checking, object/contact extraction,
    # evidence withholding, and source immutability logic remain the audited code.
    original = (ROOT / "scripts/diagnose_direct_history_replay.py").read_text()
    expected = (original.replace("direct-history", "field-history")
                .replace("fresh_direct_history_baseline", "fresh_field_history_experiment")
                .replace("live_astra_direct_history", "live_astra_field_history"))
    assert SCRIPT.read_text() == expected


def test_accepts_terminal_field_episode_and_retains_complete_transition(terminal_source):
    loaded = diagnostic.load_terminal_source(terminal_source)
    assert len(loaded["pairs"]) == 1
    assert loaded["pairs"][0][1]["action"][-1] == 1
    assert loaded["trailing_unexecuted_action"] is False
    assert len(loaded["source_hashes"]) == 5


@pytest.mark.parametrize("status,stage", [("running", "deciding"), ("budget_exceeded", "observed")])
def test_rejects_nonterminal_or_unfinalized_source(terminal_source, status, stage):
    write(terminal_source / "run_state.json", {"status": status, "stage": stage, "step": 1})
    with pytest.raises(ValueError, match="terminal|finalizing"):
        diagnostic.load_terminal_source(terminal_source)


@pytest.mark.parametrize("changes", [
    {"kind": "fresh_direct_history_baseline"}, {"mode": "offline_mock_smoke"},
    {"mode": "live_astra_direct_history"}, {"episode_index": 1},
])
def test_rejects_wrong_experiment_or_mock_identity(terminal_source, changes):
    rewrite_identity(terminal_source, **changes)
    with pytest.raises(ValueError, match="real field-history"):
        diagnostic.load_terminal_source(terminal_source)


def test_rejects_changed_source_provenance(terminal_source):
    manifest = diagnostic.read(terminal_source / "manifest.json")
    software = manifest["configuration"]["software"]
    software["source_sha256"]["schema.py"] = "0" * 64
    rewrite_identity(terminal_source, software=software)
    with pytest.raises(ValueError, match="Source package changed"):
        diagnostic.load_terminal_source(terminal_source)


def test_rejects_feedback_disagreeing_with_executed_action(terminal_source):
    path = terminal_source / "trace.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[-1]["feedback"]["normalized_action"][-1] = -1
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="Feedback contradicts"):
        diagnostic.load_terminal_source(terminal_source)


def test_goal_components_do_not_treat_gripper_close_as_success():
    components = diagnostic.on_components([0, 0, 1], [.04, 0, .99], True)
    assert components["bowl_plate_collision_contact"] is True
    assert components["on_components_conjunction"] is False
    assert components["xy_body_origin_distance_m"] == .04
