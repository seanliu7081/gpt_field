"""Queue tests use synthetic evidence and fake children; no inference or simulator."""
import hashlib
import importlib.util
import json
from pathlib import Path
import signal

import pytest

SPEC = importlib.util.spec_from_file_location("historical_queue", Path(__file__).parents[1] /
                                            "scripts/run_historical_missing_trials.py")
queue = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(queue)


def observation(scene):
    return {"observation/image": scene, "observation/wrist_image": "wrist",
            "observation/state": [0.] * 8, "prompt": "task"}


def summary(controller="hybrid", success=True):
    return {"status": "success" if success else "user_stop", "evaluation_completed": success,
            "libero_success": success, "billing": "chatgpt_subscription", "api_key_calls": 0,
            "reasoning_effort": "medium", "stop_on_model_complete": False, "seed": 7,
            "episode_index": 0, "controller_mode": controller, "control_elapsed_s": 12.,
            "timing_components": {"video_render_s": 2.}, "api_calls": 3, "steps": 15,
            "max_policy_steps": None, "max_wall_time_s": None, "max_api_calls": None}


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    root = tmp_path / "workspace"
    root.mkdir()
    monkeypatch.setattr(queue, "ROOT", root)
    monkeypatch.setattr(queue, "INITIAL_HASHES", {scene: queue.canonical_observation_hash(observation(scene))
                                                 for scene in queue.INITIAL_HASHES})
    for source in (queue.RUNNER, queue.QUEUE_SCRIPT, "src/gpt_field/stub.py"):
        path = root / source
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# frozen code\n")
    source = root / queue.RELAYOUT_SOURCE
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"immutable relayout fixture")
    monkeypatch.setattr(queue, "RELAYOUT_SHA256", hashlib.sha256(source.read_bytes()).hexdigest())
    monkeypatch.setattr(queue, "source_manifest", lambda: {"python": "fixture", "packages": {},
        "source_sha256": {"stub.py": queue.sha256((root / "src/gpt_field/stub.py").read_bytes())}})
    for _, scene, _, directory in queue.HISTORICAL:
        path = root / directory
        path.mkdir(parents=True)
        (path / "summary.json").write_text(json.dumps(summary(success=scene == "original_bowl")))
        (path / "trace.jsonl").write_text(json.dumps({"event": "reset", "observation": observation(scene)}) + "\n")
    destination = root / "queue"
    manifest = queue.prepare(destination)
    return destination, manifest


def test_exact_authorized_trials_and_commands(prepared):
    output, manifest = prepared
    assert [t["id"] for t in manifest["trials"]] == [t[0] for t in queue.TRIALS]
    for trial in manifest["trials"]:
        cmd = trial["command"]
        assert "--reasoning-effort" in cmd and cmd[cmd.index("--reasoning-effort")+1] == "medium"
        assert cmd[cmd.index("--task")+1] == trial["task_key"]
        assert cmd[cmd.index("--expected-initial-observation-sha256")+1] == queue.INITIAL_HASHES[trial["scene"]]
        assert ("--distance-field-control" in cmd) == (trial["controller"] == "hybrid")
        assert ("--historical-zero-shot-direct" in cmd) == (trial["controller"] == "direct")
        assert ("--initial-state-override" in cmd) == (trial["scene"] == "relayout_bowl")
        assert not set(cmd) & {"--demonstration", "--zero-shot-collision-free", "--stop-on-model-complete",
                               "--blocks", "--actions-per-block", "--task-bddl-override"}
    settings = manifest["settings"]
    assert settings["include_arm_joints"] is False
    assert settings["direct_action_steps"] == 5
    assert all(settings[k] is None for k in ("max_steps", "max_wall_time_s", "max_model_calls"))
    assert json.loads((output / "run_state.json").read_text())["status"] == "prepared"
    assert not any(Path(t["output"]).exists() for t in manifest["trials"])


def test_prepare_and_report_never_launch(prepared, monkeypatch):
    output, manifest = prepared
    monkeypatch.setattr(queue.subprocess, "Popen", lambda *a, **kw: pytest.fail("Must not launch"))
    result = queue.run(output, report_only=True)
    assert len(result["trials"]) == 6
    assert len(result["pairs"]) == 3
    assert not any(p["completion_comparison_eligible"] for p in result["pairs"])
    drawer = next(t for t in result["trials"] if t["id"] == "original_drawer_zero_hybrid")
    assert drawer["time_without_video_s"] == 10
    assert drawer["time_interpretation"] == "attempt_duration_only"
    with pytest.raises(FileExistsError):
        queue.run(output, prepare_only=True)


def test_input_hash_rejected_before_creating_output(prepared):
    output, _ = prepared
    (queue.ROOT / queue.RELAYOUT_SOURCE).write_bytes(b"changed")
    another = output.parent / "new"
    with pytest.raises(ValueError, match="relayout input hash"):
        queue.prepare(another)
    assert not another.exists()


def test_source_and_frozen_inputs_checked_before_launch(prepared, monkeypatch):
    output, _ = prepared
    monkeypatch.setattr(queue.subprocess, "Popen", lambda *a, **kw: pytest.fail("Must not launch"))
    source = queue.ROOT / queue.RUNNER
    source.write_text("# changed\n")
    with pytest.raises(RuntimeError, match="Source changed"):
        queue.run(output)
    source.write_text("# frozen code\n")
    (output / "inputs/initial_state_override.npy").write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        queue.run(output)


def install_child(monkeypatch, manifest, *, interrupt=None, failure=False, drift=False):
    calls = []

    class Child:
        pid = 12345

        def __init__(self, command, **kwargs):
            self.command, self.kwargs, self.signals, self.finished = command, kwargs, [], False
            calls.append(self)
            assert "OPENAI_API_KEY" not in kwargs["env"]
            assert kwargs["start_new_session"] is True

        def poll(self):
            return 0 if self.finished else None

        def send_signal(self, sig):
            self.signals.append(sig)

        def wait(self):
            trial = manifest["trials"][len(calls)-1]
            if interrupt is not None:
                signal.getsignal(interrupt)(interrupt, None)
            path = Path(trial["output"])
            path.mkdir()
            (path / "summary.json").write_text(json.dumps(summary(trial["controller"], not (failure or interrupt))))
            (path / "initial_observation.json").write_text(json.dumps(observation(trial["scene"])))
            self.finished = True
            if drift:
                (queue.ROOT / "src/gpt_field/stub.py").write_text("# changed after child\n")
            return 3 if failure or interrupt else 0

    monkeypatch.setenv("OPENAI_API_KEY", "must-not-be-forwarded")
    monkeypatch.setattr(queue.subprocess, "Popen", Child)
    return calls


def test_four_successful_children_in_order_with_adjusted_report(prepared, monkeypatch):
    output, manifest = prepared
    calls = install_child(monkeypatch, manifest)
    result = queue.run(output)
    assert [c.command for c in calls] == [t["command"] for t in manifest["trials"]]
    assert len(list(output.glob("*.log"))) == 4
    state = json.loads((output / "run_state.json").read_text())
    assert state["status"] == "complete" and state["completed"] == [t["id"] for t in manifest["trials"]]
    pairs = {p["scene"]: p for p in result["pairs"]}
    assert pairs["relayout_bowl"]["completion_comparison_eligible"]
    assert pairs["original_bowl"]["completion_comparison_eligible"]
    assert pairs["relayout_bowl"]["direct_minus_hybrid_s"] == 0
    assert pairs["original_bowl"]["comparison_type"] == "retrospective_hybrid_reference"
    assert not pairs["original_drawer"]["completion_comparison_eligible"]
    assert pairs["original_drawer"]["direct_over_hybrid_time_ratio"] is None
    with pytest.raises(RuntimeError, match="fresh prepared"):
        queue.run(output)


@pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGTERM])
def test_signal_waits_for_child_cleanup_and_never_runs_next(prepared, monkeypatch, sig):
    output, manifest = prepared
    before = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
    calls = install_child(monkeypatch, manifest, interrupt=sig)
    result = queue.run(output)
    assert len(calls) == 1 and calls[0].finished
    assert calls[0].signals == [signal.SIGINT]
    state = json.loads((output / "run_state.json").read_text())
    assert state["status"] == "interrupted" and state["interrupt_signal"] == sig
    assert all(signal.getsignal(s) is h for s, h in before.items())
    assert not any(p["completion_comparison_eligible"] for p in result["pairs"])


def test_error_stops_without_starting_next(prepared, monkeypatch):
    output, manifest = prepared
    calls = install_child(monkeypatch, manifest, failure=True)
    queue.run(output)
    assert len(calls) == 1
    assert json.loads((output / "run_state.json").read_text())["status"] == "incomplete_or_error"


def test_code_drift_between_trials_stops(prepared, monkeypatch):
    output, manifest = prepared
    calls = install_child(monkeypatch, manifest, drift=True)
    with pytest.raises(RuntimeError, match="Source changed"):
        queue.run(output)
    assert len(calls) == 1
    assert json.loads((output / "run_state.json").read_text())["status"] == "queue_error"


@pytest.mark.parametrize("bad", ["wrong_initial", "invalid_time", "api_key", "model_completion"])
def test_success_status_alone_never_enables_speedup(prepared, bad):
    output, manifest = prepared
    trial = manifest["trials"][0]
    path = Path(trial["output"])
    path.mkdir()
    data = summary("direct")
    obs = observation(trial["scene"])
    if bad == "wrong_initial":
        obs["observation/state"][0] = 1
    if bad == "invalid_time":
        data["timing_components"]["video_render_s"] = 100
    if bad == "api_key":
        data["api_key_calls"] = 1
    if bad == "model_completion":
        data["libero_success"] = False
    (path / "summary.json").write_text(json.dumps(data))
    (path / "initial_observation.json").write_text(json.dumps(obs))
    result = queue.summarize(manifest, output)
    pair = result["pairs"][0]
    assert not pair["completion_comparison_eligible"]
    assert pair["hybrid_time_reduction_percent"] is None
