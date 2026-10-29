"""Recorded Direct comparison replay: no live model, credentials or simulator."""

import copy
from dataclasses import replace
import importlib.util
import json
from pathlib import Path

import pytest
import numpy as np

from gpt_field.demonstration import load_demonstration
from gpt_field.distance_tasks import resolve_distance_task
from gpt_field.experiment import digest
from gpt_field.libero import validate_task_bddl_override
from test_demonstration import write_bundle
from test_demo_audit import demo_bundle, demo_fixture, mock_override_metadata, mock_override_reset
from test_distance_audit import rewrite, rewrite_trace
from test_field_history import FakeTransport, envelope
from test_field_history_runner import FakeRecorder, fake_simulator, observation, rows
from test_hybrid_subscription_runner import runner, sim_config

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("audit_demo_direct_tests", ROOT / "scripts/audit_demo_direct_control.py")
auditor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(auditor)


def direct_fixture(tmp_path, observation, *, interrupted=False, real_video=False, task_key="drawer_bowl", episode_index=0,
                   override_path=None, bddl_path=None, original_bddl_path=None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    source = write_bundle(tmp_path, demo_bundle(task_key))
    reference = load_demonstration(source, task_key=task_key)
    task = resolve_distance_task(task_key)
    obs = replace(observation, prompt=task.prompt)
    base, instances = fake_simulator(obs, success_at=None if interrupted else 11)

    class DrawerSimulator(base):
        def __init__(self, config):
            super().__init__(config)
            mock_override_metadata(self)

        def list_tasks(self):
            return [{"task_id": task.task_id, "prompt": task.prompt}]

        def reset(self, task_id, selected_episode):
            assert (task_id, selected_episode) == (task.task_id, episode_index)
            initial = super().reset(0, 0)
            self.status.update(task_id=task_id, episode_index=selected_episode)
            mock_override_reset(self, original_bddl_path)
            return initial

    responses = []
    phases = ("handle_grasp", "drawer_open", "handle_release") if task_key == "drawer_bowl" else ("approach", "grasp", "release")
    for index, phase in enumerate(phases):
        raw = {"actions": [[.1 * (step + 1), 0., 0., 0., 0., 0., -1.] for step in range(5)],
               "target_region": "drawer_handle" if task_key == "drawer_bowl" else ("plate" if phase == "release" else "black_bowl"),
               "routing_reason": "observable task interaction",
               "control_context": {"phase": phase, "visual_reference": None}}
        responses.append(envelope(raw, id=f"direct-audit-{index}", model="gpt-6-astra", reasoning={"effort": "medium"}))
    if interrupted:
        responses[1] = KeyboardInterrupt("fixture interrupted next Direct request")
    transport = FakeTransport(*responses)
    run = tmp_path / "direct_run"
    config = sim_config(suite=task.suite)
    if override_path is not None:
        config = replace(config, initial_state_override=str(override_path))
    if bddl_path is not None:
        config = replace(config, task_bddl_override=str(bddl_path))
    summary = runner.execute_episode(run, config=runner.configuration(), sim_config=config,
        token_provider=lambda: "fixture-oauth", transport=transport, sim_factory=DrawerSimulator,
        recorder_factory=None if real_video else FakeRecorder, task_key=task_key, demonstration_path=source,
        episode_index=episode_index,
        controller_mode="direct", reasoning_effort="medium", execution_mode="fixed", feedback_history="block_summary",
        centered_placement_guidance=True, grasp_settle_guidance=True)
    assert summary["status"] == ("user_stop" if interrupted else "success"), summary.get("error")
    assert len(instances) == 1
    return run, reference, transport


@pytest.fixture
def skip_video(monkeypatch):
    def check(run, summary, actions, executed, feedback, audit):
        audit.notes.append("Synthetic fixture recorder omitted decoder in this isolated replay test.")
        return {"synthetic_mock_recorder": True}, 0
    monkeypatch.setattr(auditor._legacy, "_video_audit", check)


@pytest.mark.parametrize("task_key", ["drawer_bowl", "bowl_plate"])
def test_reconstructs_pure_direct_five_actions_exact_demo_and_history(tmp_path, observation, skip_video, task_key):
    run, reference, transport = direct_fixture(tmp_path, observation, task_key=task_key)
    report = auditor.audit_run(run, tmp_path / "audit")
    assert report["passed"], report["issues"]
    assert report["executed_actions"] == report["feedback_records"] == 11
    assert report["replayed_recorded_requests"] == report["accepted_recorded_responses"] == 3
    assert report["demonstration"] == reference.metadata()
    assert report["distance_task"] == task_key
    assert report["reconstructed_local_plans"] == 0 and not report["distance_field_control"]
    assert report["synthetic_test_fixture"] and not report["real_task_completion_verified"]
    assert report["live_model_calls_by_audit"] == report["simulator_actions_by_audit"] == 0
    assert report["reference_comparison"] is None
    summary = json.loads((run / "summary.json").read_text())
    assert report["recorded_control_excluding_video_s"] == summary["control_elapsed_s"] - summary["timing_components"]["video_render_s"]
    assert "not a no-video rerun" in report["adjusted_timing_note"]
    for index, payload in enumerate(transport.calls):
        assert payload["input"][0] == reference.message()
        assert sum(item == reference.message() for item in payload["input"]) == 1
        live_images = sum(part.get("type") == "input_image" for item in payload["input"][1:]
                          for part in item.get("content", []) if isinstance(part, dict))
        assert live_images == min(index + 1, 2) * 2


@pytest.mark.parametrize("tamper", ["action", "field_plan", "prefix", "hidden_observation", "completion"])
@pytest.mark.parametrize("task_key", ["drawer_bowl", "bowl_plate"])
def test_tampered_direct_evidence_is_rejected(tmp_path, observation, skip_video, tamper, task_key):
    run, _, _ = direct_fixture(tmp_path, observation, task_key=task_key)
    trace = rows(run / "trace.jsonl")
    if tamper == "prefix":
        path = sorted((run / "requests").glob("*.request.json"))[1]
        request = json.loads(path.read_text())
        request["payload"]["input"].insert(0, copy.deepcopy(request["payload"]["input"][0]))
        rewrite(path, request)
    else:
        if tamper == "action":
            next(row for row in trace if row["event"] == "action")["action"][0] += .01
        elif tamper == "field_plan":
            next(row for row in trace if row["event"] == "action")["diagnostics"]["plan"] = {"status": "move"}
        elif tamper == "hidden_observation":
            next(row for row in trace if row["event"] == "executed")["observation"]["object_pose"] = [0., 0., 0.]
        else:
            for row in trace:
                if row["event"] == "executed":
                    row["success"] = row["sim_status"]["success"] = False
        rewrite_trace(run, trace)
    report = auditor.audit_run(run, tmp_path / "audit")
    assert not report["passed"]
    expected = {"action": "exact unchanged model action replay", "field_plan": "no field, motion planning",
                "prefix": "one exact immutable external demonstration", "hidden_observation": "four allowed fields",
                "completion": "success status requires actual evaluator evidence"}
    assert any(expected[tamper] in issue for issue in report["issues"]), report["issues"]


@pytest.mark.parametrize("task_key", ["drawer_bowl", "bowl_plate"])
def test_interrupted_next_request_is_incomplete_not_demo_success(tmp_path, observation, skip_video, task_key):
    run, _, _ = direct_fixture(tmp_path, observation, interrupted=True, task_key=task_key)
    report = auditor.audit_run(run, tmp_path / "audit")
    assert report["passed"], report["issues"]
    assert report["status"] == "user_stop" and not report["evaluation_completed"]
    assert report["executed_actions"] == 5 and report["trace_complete"]
    assert report["replayed_recorded_requests"] == 2 and report["accepted_recorded_responses"] == 1
    assert report["usage_totals"]["unknown_usage_calls"] == 1


@pytest.mark.parametrize("tamper", [None, "initial_state", "guidance", "demo"])
@pytest.mark.parametrize("task_key", ["drawer_bowl", "bowl_plate"])
def test_reference_comparison_checks_actual_reset_demo_and_shared_task_guidance(
        tmp_path, observation, skip_video, tamper, task_key):
    hybrid_parent = tmp_path / "hybrid"
    hybrid_parent.mkdir()
    reference_run, _, _ = demo_fixture(hybrid_parent, observation, task_key=task_key)
    run, _, _ = direct_fixture(tmp_path / "direct", observation, task_key=task_key)
    if tamper == "initial_state":
        trace = rows(reference_run / "trace.jsonl")
        measured = auditor.PolicyObservation.from_dict(trace[0]["observation"])
        state = list(measured.state)
        state[0] += .001
        trace[0]["observation"] = replace(measured, state=tuple(state)).to_dict()
        rewrite_trace(reference_run, trace)
    elif tamper == "guidance":
        path = sorted((reference_run / "requests").glob("*.request.json"))[1]
        request = json.loads(path.read_text())
        request["payload"]["instructions"] = "changed common task guidance"
        rewrite(path, request)
    elif tamper == "demo":
        with (reference_run / "demonstration.json").open("a") as stream:
            stream.write("\n")
    report = auditor.audit_run(run, tmp_path / "audit", reference_run=reference_run)
    assert report["passed"] is (tamper is None), report["issues"]
    assert report["reference_comparison"]["matched_checks_passed"] is (tamper is None)
    if tamper:
        expected = {"initial_state": "matched actual initial", "guidance": "shares verbatim task guidance",
                    "demo": "matched exact demonstration artifact bytes"}
        assert any(expected[tamper] in issue for issue in report["issues"]), report["issues"]


def test_actual_direct_video_decodes_and_matches_all_events(tmp_path, observation):
    run, _, _ = direct_fixture(tmp_path, observation, real_video=True)
    report = auditor.audit_run(run, tmp_path / "audit")
    assert report["passed"], report["issues"]
    assert report["event_frames"] > 0


@pytest.mark.parametrize("episode_index", [1, 49])
def test_pair_replays_matching_nonzero_initial_state_without_changing_demo(
        tmp_path, observation, skip_video, episode_index):
    hybrid_parent = tmp_path / "hybrid"
    hybrid_parent.mkdir()
    hybrid, ref, _ = demo_fixture(hybrid_parent, observation, task_key="bowl_plate", episode_index=episode_index)
    direct, direct_ref, _ = direct_fixture(tmp_path / "direct", observation, task_key="bowl_plate", episode_index=episode_index)
    assert ref.metadata() == direct_ref.metadata()
    hybrid_report = auditor._helpers.audit_run(hybrid, tmp_path / "hybrid_audit")
    assert hybrid_report["passed"], hybrid_report["issues"]
    report = auditor.audit_run(direct, tmp_path / "direct_audit", reference_run=hybrid)
    assert report["passed"], report["issues"]
    assert hybrid_report["episode_index"] == report["episode_index"] == episode_index
    assert report["reference_comparison"]["matched_checks_passed"]


@pytest.mark.parametrize("where", ["summary", "reset", "sim_reset", "progress"])
def test_nonzero_reset_bookkeeping_must_agree_with_requested_layout(
        tmp_path, observation, skip_video, where):
    run, _, _ = direct_fixture(tmp_path, observation, task_key="bowl_plate", episode_index=1)
    if where in ("summary", "progress"):
        path = run / ("summary.json" if where == "summary" else "run_state.json")
        value = json.loads(path.read_text())
        value["episode_index"] = 0
        rewrite(path, value)
    else:
        trace = rows(run / "trace.jsonl")
        target = trace[0] if where == "reset" else trace[0]["sim_status"]
        target["episode_index"] = 0
        rewrite_trace(run, trace)
    report = auditor.audit_run(run, tmp_path / "audit")
    assert not report["passed"]
    assert any("episode_index" in issue or "initial-state index" in issue for issue in report["issues"])


@pytest.mark.parametrize("tamper", [None, "artifact", "sim_reset", "reference_layout"])
def test_custom_layout_bytes_and_reset_provenance_match_across_arms(
        tmp_path, observation, skip_video, tamper):
    source = tmp_path / "layout.npy"
    np.save(source, np.arange(145, dtype=np.float64))
    hybrid_parent = tmp_path / "hybrid"
    hybrid_parent.mkdir()
    hybrid, _, _ = demo_fixture(hybrid_parent, observation, task_key="bowl_plate", override_path=source)
    direct, _, _ = direct_fixture(tmp_path / "direct", observation, task_key="bowl_plate",
                                  override_path=hybrid / "initial_state_override.npy")
    if tamper == "artifact":
        np.save(direct / "initial_state_override.npy", np.arange(145, dtype=np.float64) + .01)
    elif tamper == "sim_reset":
        trace = rows(direct / "trace.jsonl")
        trace[0]["sim_status"]["initial_state_override_sha256"] = "0" * 64
        rewrite_trace(direct, trace)
    elif tamper == "reference_layout":
        np.save(hybrid / "initial_state_override.npy", np.arange(145, dtype=np.float64) + .01)
    report = auditor.audit_run(direct, tmp_path / "audit", reference_run=hybrid)
    assert report["passed"] is (tamper is None), report["issues"]
    if tamper is None:
        hybrid_report = auditor._helpers.audit_run(hybrid, tmp_path / "hybrid_audit")
        assert hybrid_report["passed"], hybrid_report["issues"]
        assert report["benchmark_variant"] == "custom_layout"
        assert report["initial_state_override"] == hybrid_report["initial_state_override"]
    else:
        expected = {"artifact": "custom initial-state bytes", "sim_reset": "confirms selected frozen custom state",
                    "reference_layout": "matched exact custom initial-state bytes"}
        assert any(expected[tamper] in issue for issue in report["issues"]), report["issues"]


def scene_files(tmp_path):
    original = tmp_path / "official.bddl"
    original.write_text("""(define (problem LIBERO_Tabletop_Manipulation)
      (:domain robosuite)
      (:language pick up the black bowl between the plate and the ramekin and place it on the plate)
      (:fixtures main_table - table)
      (:objects akita_black_bowl_1 - akita_black_bowl plate_1 - plate)
      (:obj_of_interest akita_black_bowl_1 plate_1)
      (:goal (And (On akita_black_bowl_1 plate_1))))\n""")
    custom = tmp_path / "obstacles.bddl"
    custom.write_text(original.read_text().replace("plate_1 - plate)", "plate_1 - plate wooden_cabinet_2 - wooden_cabinet)"))
    return original, custom


@pytest.mark.parametrize("tamper", [None, "artifact", "goal", "target_class", "original", "sim_reset", "reference_scene"])
def test_custom_scene_task_preservation_frozen_provenance_and_pair_matching(
        tmp_path, observation, skip_video, tamper):
    source = tmp_path / "layout.npy"
    np.save(source, np.arange(152, dtype=np.float64))  # More objects may change the model-state length.
    original, custom = scene_files(tmp_path)
    hybrid_parent = tmp_path / "hybrid"
    hybrid_parent.mkdir()
    hybrid, _, _ = demo_fixture(hybrid_parent, observation, task_key="bowl_plate", override_path=source,
                               bddl_path=custom, original_bddl_path=original)
    direct, _, _ = direct_fixture(tmp_path / "direct", observation, task_key="bowl_plate",
        override_path=hybrid / "initial_state_override.npy", bddl_path=hybrid / "task_override.bddl",
        original_bddl_path=original)
    if tamper in ("artifact", "goal", "target_class"):
        artifact = direct / "task_override.bddl"
        text = artifact.read_text()
        changed = {"artifact": text + "; changed bytes\n",
                   "goal": text.replace("(On akita_black_bowl_1 plate_1)", "(On plate_1 akita_black_bowl_1)"),
                   "target_class": text.replace("akita_black_bowl_1 - akita_black_bowl", "akita_black_bowl_1 - plate")}
        artifact.write_text(changed[tamper])
        if tamper in ("goal", "target_class"):
            # Even internally rehashed metadata cannot conceal a changed task.
            updated = validate_task_bddl_override(artifact)
            manifest = json.loads((direct / "manifest.json").read_text())
            for filename, config in (("manifest.json", manifest["configuration"]),
                                     ("summary.json", json.loads((direct / "summary.json").read_text()))):
                for field in ("task_bddl_override", "task_bddl_override_source"):
                    config[field].update({key: updated[key] for key in ("sha256", "byte_count")})
                if filename == "manifest.json":
                    manifest["identity_hash"] = digest(config)
                    rewrite(direct / filename, manifest)
                else:
                    rewrite(direct / filename, config)
    elif tamper == "original":
        (direct / "original_task.bddl").write_text(original.read_text() + "; changed archive\n")
    elif tamper == "sim_reset":
        trace = rows(direct / "trace.jsonl")
        trace[0]["sim_status"]["task_bddl_override_sha256"] = "0" * 64
        rewrite_trace(direct, trace)
    elif tamper == "reference_scene":
        with (hybrid / "task_override.bddl").open("a") as stream:
            stream.write("; different reference scene\n")
    report = auditor.audit_run(direct, tmp_path / "audit", reference_run=hybrid)
    assert report["passed"] is (tamper is None), report["issues"]
    if tamper is None:
        hybrid_report = auditor._helpers.audit_run(hybrid, tmp_path / "hybrid_audit")
        assert hybrid_report["passed"], hybrid_report["issues"]
        assert report["benchmark_variant"] == "custom_scene"
        assert report["task_bddl_override"] == hybrid_report["task_bddl_override"]
        assert report["task_bddl_override"]["task_contract"]["target_assets"] == {
            "akita_black_bowl_1": "akita_black_bowl", "plate_1": "plate"}
    else:
        expected = {"artifact": "custom BDDL bytes match manifest", "goal": "preserves original task language, goal",
                    "target_class": "preserves original task language, goal", "original": "anchors archived official BDDL identity",
                    "sim_reset": "confirms selected frozen custom scene", "reference_scene": "matched exact scene artifact"}
        assert any(expected[tamper] in issue for issue in report["issues"]), report["issues"]
