"""Offline v2 orchestration tests; all child processes, observations and inputs are fake."""
import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import signal
import sys

import pytest

SCRIPTS = Path(__file__).parents[1] / "scripts"


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


prep = load_script("prepare_visual_field_v2_trials")
queue = load_script("run_visual_field_v2_trials")


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def digest(data):
    return hashlib.sha256(data).hexdigest()


def source_fixture(tmp_path):
    project = tmp_path / "project"
    original = project / "runs/visual_field_matrix_v1"
    scenes, artifacts = [], []
    for name in prep.SCENES:
        task = "drawer_bowl" if name == "drawer_original" else "bowl_plate"
        scene = {"id": name, "task": task, "suite": "libero_goal" if name == "drawer_original" else "libero_spatial",
                 "task_id": 3 if name == "drawer_original" else 0,
                 "task_prompt": "fixture task " + task, "seed": 7, "episode_index": 0,
                 "category": "fixture", "source_run": str(project / "archived" / name),
                 "initial_state": None, "initial_state_sha256": None, "bddl": None}
        for key, filename in (("official_bddl", "original_task.bddl"), ("official_initial_state", "official.pruned_init")):
            data = (name + key).encode()
            live = project / "official" / name / filename
            frozen = original / "inputs" / name / filename
            for path in (live, frozen):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
            scene[key] = {"path": str(live), "sha256": digest(data)}
            artifacts.append({"path": str(frozen), "sha256": digest(data)})
        scene["actual_bddl_sha256"] = scene["official_bddl"]["sha256"]
        for key, filename in (("initial_state", "initial_state.npy"), ("bddl", "task_override.bddl")):
            if (key == "initial_state" and name in ("bowl_relayout", "bowl_obstacles")) or (key == "bddl" and name == "bowl_obstacles"):
                data = (name + key).encode()
                path = original / "inputs" / name / filename
                path.write_bytes(data)
                scene[key] = str(path)
                scene["initial_state_sha256" if key == "initial_state" else "actual_bddl_sha256"] = digest(data)
                artifacts.append({"path": str(path), "sha256": digest(data)})
        image = (name + "fake png, never decoded by the pure preparation").encode()
        obs = {"observation/image": "data:image/png;base64," + base64.b64encode(image).decode(),
               "observation/wrist_image": "data:image/png;base64," + base64.b64encode(image).decode(),
               "observation/state": [0.] * 8, "prompt": scene["task_prompt"]}
        reset = {"event": "reset", "observation": obs, "sim_status": {"evaluator_only": True}}
        trace = Path(scene["source_run"]) / "trace.jsonl"
        trace.parent.mkdir(parents=True)
        trace.write_text(json.dumps(reset) + "\n")
        full = {**obs, "observation/joint_position": [0.1] * 7, "observation/joint_velocity": [0.2] * 7}
        for repeat in (1, 2):
            status = {"task_id": scene["task_id"], "episode_index": 0,
                      "bddl_sha256": scene["actual_bddl_sha256"],
                      "initial_state_sha256": scene["official_initial_state"]["sha256"],
                      "initial_state_override_sha256": scene["initial_state_sha256"],
                      "collision": {"detected": False}}
            record = {"scene": name, "initial_status": status,
                      "configuration": {"seed": 7, "include_arm_joints": True, "collision_check_mode": "task_manipulation_v1"},
                      "joint_position": full["observation/joint_position"], "joint_velocity": full["observation/joint_velocity"],
                      "observation_sha256": digest(prep.canonical(full)), "rgb": {}}
            for view in ("external", "wrist"):
                path = original / "validation" / f"{name}_reset_{repeat}_{view}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(image)
                record["rgb"][view] = {"artifact": path.name, "sha256": digest(image)}
            write(original / "validation" / f"{name}_reset_{repeat}.json", record)
        scenes.append(scene)
        write(project / "runs" / prep.DIRECT_REFERENCES[name] / "summary.json", terminal("user_stop"))
    # A missing demonstration artifact proves preparation does not open/copy it.
    artifacts.append({"path": str(original / "DO_NOT_OPEN_demonstration.json"), "sha256": "0" * 64})
    write(original / "matrix.json", {"schema_version": "visual-field-matrix-v1", "scenes": scenes,
                                     "frozen_input_artifacts": artifacts})
    for path in (project / "src/gpt_field/policy.py", project / "scripts/evaluate_visual_field_subscription.py",
                 project / "scripts/run_visual_field_v2_trials.py"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# frozen fixture\n")
    return project, original / "matrix.json"


def terminal(status="success"):
    return {"status": status, "task_success": status == "success", "libero_success": status == "success",
            "method_version": "v2", "controller": "hybrid", "shots": 0,
            "completion_time_eligible": status == "success", "collision": {"detected": status == "collision_failure",
                "enabled": True, "mode": "task_manipulation_v1"},
            "steps": 0, "field_actions": 0, "direct_actions": 0, "control_elapsed_s": 12.,
            "timing_components": {"video_render_s": 2.}, "control_without_video_s": 10.,
            "budget": {"api_calls": 1}, "attempts": [], "recovery_events": [], "cleanup_errors": []}


@pytest.fixture
def prepared(tmp_path):
    project, source = source_fixture(tmp_path)
    output = project / "runs/v2"
    plan = prep.prepare(project, output, source)
    return project, output, plan


def test_prepare_only_four_zero_hybrid_and_preserves_exact_reset(prepared):
    project, output, plan = prepared
    assert len(plan["trials"]) == 4 and plan["new_direct_trials"] == 0
    assert [t["scene"] for t in plan["trials"]] == list(prep.SCENES)
    assert all(t["demo"] is None and t["controller"] == "hybrid" for t in plan["trials"])
    for t in plan["trials"]:
        args = t["runner_arguments"]
        assert args[args.index("--method-version") + 1] == "v2"
        assert "--demo" not in args and "--expected-initial" in args
        assert not any("max-" in a or "timeout" in a for a in args)
        obs = prep.read_json(t["expected_initial"])
        assert set(obs) == prep.OBSERVATION_KEYS and obs["observation/joint_velocity"] == [.2] * 7
    assert not list(output.rglob("*demonstration*"))
    assert not (output / "frozen_source").exists()  # freeze only after root's final validation
    assert all(plan["settings"][key] is None for key in ("max_steps", "max_wall_time_s", "max_model_calls"))
    assert plan["settings"]["model_completion_stopping"] is False
    with pytest.raises(FileExistsError):
        prep.prepare(project, output)


@pytest.mark.parametrize("tamper", ["qdot", "reset_collision", "custom_state", "official_source", "image"])
def test_preparation_refuses_changed_evidence_without_creating_output(tmp_path, tamper):
    project, source = source_fixture(tmp_path)
    matrix = prep.read_json(source)
    record_path = source.parent / "validation/bowl_original_reset_1.json"
    record = prep.read_json(record_path)
    if tamper == "qdot":
        record["joint_velocity"][0] = 0
        write(record_path, record)
    elif tamper == "reset_collision":
        record["initial_status"]["collision"]["detected"] = True
        write(record_path, record)
    elif tamper == "custom_state":
        Path(matrix["scenes"][1]["initial_state"]).write_bytes(b"tampered")
    elif tamper == "official_source":
        Path(matrix["scenes"][0]["official_bddl"]["path"]).write_bytes(b"tampered")
    else:
        (source.parent / "validation" / record["rgb"]["external"]["artifact"]).write_bytes(b"tampered")
    with pytest.raises(ValueError):
        prep.prepare(project, project / "runs/v2", source)
    assert not (project / "runs/v2").exists()


def fake_popen_factory(plan, outcomes, *, after_wait=None, signal_stop=None):
    calls = []

    def launch(args, **kwargs):
        index = len(calls)
        trial = plan["trials"][index]
        assert args[1:] == trial["runner_arguments"]
        calls.append((args, kwargs))
        out = Path(trial["output"])
        out.mkdir(parents=True)

        class Child:
            pid = 1234 + index
            signals = []

            def poll(self):
                return None

            def send_signal(self, signum):
                self.signals.append(signum)

            def wait(self):
                status = outcomes[index]
                if signal_stop:
                    signal_stop.request(signal.SIGTERM)
                    signal_stop.request(signal.SIGINT)
                    assert self.signals == [signal.SIGINT]
                    status = "user_stop"
                write(out / "summary.json", terminal(status))
                write(out / "initial_observation.json", prep.read_json(trial["expected_initial"]))
                if after_wait:
                    after_wait(index, out)
                return 0 if status in queue.ALLOWED_TERMINAL else 3

        return Child()

    return launch, calls


def test_serial_queue_success_collision_then_success_excludes_api_key(prepared, monkeypatch):
    project, output, plan = prepared
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-reach-child")
    launch, calls = fake_popen_factory(plan, ["success", "collision_failure", "success", "success"])
    result = queue.run(output / "matrix.json", popen=launch)
    assert len(calls) == 4 and result["queue"]["status"] == "completed"
    assert result["comparison"]["successful"] == 3 and result["comparison"]["collision_failures"] == 1
    assert all("OPENAI_API_KEY" not in call[1]["env"] for call in calls)
    assert all(call[1]["start_new_session"] for call in calls)
    assert all(row["speed_ratio"] is None for row in result["comparison"]["trials"])
    assert (output / "frozen_source/src/gpt_field/policy.py").read_text() == "# frozen fixture\n"


@pytest.mark.parametrize("status", ["user_stop", "subscription_error", "infrastructure_error"])
def test_incomplete_pauses_without_retry_or_next(prepared, status):
    _, output, plan = prepared
    launch, calls = fake_popen_factory(plan, [status])
    result = queue.run(output / "matrix.json", popen=launch)
    assert len(calls) == 1 and result["queue"]["status"] == "paused"
    assert result["comparison"]["not_started"] == 3 and result["comparison"]["incomplete"] == 1
    assert result["comparison"]["trials"][0]["time_kind"] == "attempt"
    with pytest.raises(ValueError, match="no implicit retry"):
        queue.run(output / "matrix.json", popen=launch)
    assert len(calls) == 1


def test_signals_forward_once_wait_cleanup_and_never_launch_next(prepared):
    _, output, plan = prepared
    stop = queue.QueueStop()
    launch, calls = fake_popen_factory(plan, ["success"], signal_stop=stop)
    result = queue.run(output / "matrix.json", popen=launch, stop=stop)
    assert len(calls) == 1 and result["queue"]["stop_requested"] is True
    assert (Path(plan["trials"][0]["output"]) / "summary.json").exists()
    assert result["comparison"]["trials"][0]["status"] == "user_stop"


def test_source_changes_pause_before_next_inference(prepared):
    project, output, plan = prepared

    def modify(index, _):
        (project / "src/gpt_field/policy.py").write_text("# changed\n")

    launch, calls = fake_popen_factory(plan, ["success"], after_wait=modify)
    with pytest.raises(ValueError, match="Source or prepared matrix changed"):
        queue.run(output / "matrix.json", popen=launch)
    assert len(calls) == 1 and prep.read_json(output / "queue_state.json")["status"] == "paused"


@pytest.mark.parametrize("tamper", ["extra_direct", "demo", "cap", "input", "expected_reset"])
def test_tampered_protocol_stops_before_child(prepared, tamper):
    _, output, plan = prepared
    if tamper == "extra_direct":
        plan["trials"].append(dict(plan["trials"][0], controller="direct"))
    elif tamper == "demo":
        plan["trials"][0]["demo"] = "/fake/demo.json"
    elif tamper == "cap":
        plan["trials"][0]["runner_arguments"] += ["--max-steps", "100"]
    elif tamper == "input":
        Path(plan["trials"][1]["initial_state"]).write_bytes(b"changed")
    else:
        Path(plan["trials"][0]["expected_initial"]).write_text("{}")
    write(output / "matrix.json", plan)
    with pytest.raises(ValueError):
        queue.run(output / "matrix.json", popen=lambda *a, **kw: pytest.fail("unexpected child"))


def test_actual_reset_mismatch_prevents_next_trial(prepared):
    _, output, plan = prepared

    def modify(_, out):
        write(out / "initial_observation.json", {"wrong": "reset"})

    launch, calls = fake_popen_factory(plan, ["success"], after_wait=modify)
    result = queue.run(output / "matrix.json", popen=launch)
    assert len(calls) == 1 and result["queue"]["reason"] == "initial_observation_mismatch"
    assert result["comparison"]["trials"][0]["initial_observation_exact_match"] is False


def test_model_or_false_summary_completion_never_counts():
    row = terminal("success")
    row["libero_success"] = False
    row["model_task_complete"] = True
    assert queue.qualified_outcome(row) == "invalid_terminal_record"


@pytest.mark.parametrize("change", [{"method_version": "v1"}, {"shots": 1}, {"controller": "direct"},
                                    {"collision": {"detected": False, "enabled": False}}])
def test_terminal_success_requires_v2_zero_hybrid_contact_contract(change):
    assert queue.qualified_outcome(dict(terminal(), **change)) == "invalid_terminal_record"
    row = terminal("user_stop")
    row["task_success"] = True
    assert queue.qualified_outcome(row) == "invalid_terminal_record"


def test_activated_rejected_bootstrap_count_only_acknowledged_commands(tmp_path):
    rows = []
    for step, status, accepted in ((0, "visual_reference_unavailable", False), (1, "bootstrap_empirical_image_checked", True),
                                    (2, "visual_distance_descent", True), (3, "visual_distance_descent", True)):
        rows.append({"event": "action", "step": step, "diagnostics": {"controller": "visual_field",
                     "selection": {"status": status, "accepted": accepted}}})
        if step != 3:
            rows.append({"event": "executed", "step": step})
    (tmp_path / "trace.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))
    result = queue.activation_counts(tmp_path, {"steps": 3, "field_actions": 3})
    assert result["local_servo_actions"] == result["field_rejected_actions"] == result["bootstrap_actions"] == 1
    assert result["acknowledged_actions"] == 3
    with pytest.raises(ValueError, match="activation counter"):
        queue.activation_counts(tmp_path, {"steps": 3, "field_actions": 3, "local_servo_actions": 2})


def test_attachment_gated_proposals_are_neither_servo_nor_bootstrap(tmp_path):
    rows = []
    for step, status in enumerate(("visual_distance_descent", "bootstrap_empirical_image_checked")):
        rows.extend([{"event": "action", "step": step, "diagnostics": {
            "controller": "visual_field", "attachment_gate": "payload_unverified",
            "selection": {"status": status, "accepted": True}}},
            {"event": "executed", "step": step}])
    (tmp_path / "trace.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))
    counts = queue.activation_counts(tmp_path, {"steps": 2, "field_actions": 2, "local_servo_actions": 0})
    assert counts["local_servo_actions"] == counts["bootstrap_actions"] == 0
    assert counts["attachment_gate_actions"] == 2


def test_reporting_never_scores_pending_and_no_speed_for_any_reference(prepared):
    _, output, plan = prepared
    out = Path(plan["trials"][0]["output"])
    write(out / "run_state.json", {"status": "running", "step": 7})
    result = queue.run(output / "matrix.json", report_only=True)
    assert result["trials"][0]["status"] == "running" and result["not_started"] == 3
    assert result["incomplete"] == 0
    assert all(r["speed_ratio"] is None for r in result["trials"])
    assert queue.time_without_video(terminal()) == 10.
    with pytest.raises(ValueError, match="inconsistent"):
        queue.time_without_video(dict(terminal(), control_without_video_s=99))
