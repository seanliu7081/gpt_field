"""One external observable demonstration is input, never a scored live episode."""

import copy
from dataclasses import replace
import hashlib
import json

import pytest

from gpt_field.demonstration import load_demonstration
from gpt_field.distance_subscription import distance_prompt_version
from gpt_field.distance_tasks import resolve_distance_task
from gpt_field.experiment import digest
from gpt_field.libero import validate_initial_state_override, validate_task_bddl_override, validate_task_bddl_contract
from gpt_field.observation import PolicyObservation
from test_demonstration import make_bundle, write_bundle
from test_distance_audit import auditor, rewrite, skip_fake_video
from test_distance_policy import intent
from test_distance_runner import SETTINGS
from test_field_history import FakeTransport, envelope
from test_field_history_runner import FakeRecorder, fake_simulator, observation, rows
from test_hybrid_subscription_runner import runner, sim_config


def demo_bundle(task_key="drawer_bowl"):
    bundle = make_bundle()
    task = resolve_distance_task(task_key)
    bundle.update(task_key=task.key, task_prompt=task.prompt)
    bundle["source"].update(suite=task.suite, task_id=task.task_id)
    if task_key == "bowl_plate":
        bundle["source"]["file_name"] = "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo.hdf5"
    return bundle


def mock_override_metadata(simulator):
    if simulator.config.initial_state_override is not None:
        simulator.metadata.update(initial_state_override=validate_initial_state_override(simulator.config.initial_state_override),
            benchmark_variant="LIBERO task with custom initial layout (explicit checkout)")
    if simulator.config.task_bddl_override is not None:
        simulator.metadata.update(task_bddl_override=validate_task_bddl_override(simulator.config.task_bddl_override),
            benchmark_variant="LIBERO task with custom scene and initial layout (explicit checkout)")


def mock_override_reset(simulator, original_bddl_path=None):
    if simulator.config.initial_state_override is not None:
        metadata = simulator.metadata["initial_state_override"]
        simulator.status.update(initial_state_source="custom_override", initial_state_override_file=metadata["path"],
                                initial_state_override_sha256=metadata["sha256"])
    if simulator.config.task_bddl_override is not None:
        metadata = simulator.metadata["task_bddl_override"]
        simulator.status.update(task_bddl_override_file=metadata["path"], task_bddl_override_sha256=metadata["sha256"],
            bddl_sha256=metadata["sha256"], original_bddl_file=str(original_bddl_path.resolve()),
            original_bddl_sha256=hashlib.sha256(original_bddl_path.read_bytes()).hexdigest(),
            custom_task_contract=validate_task_bddl_contract(original_bddl_path, metadata["path"]))


def demo_fixture(tmp_path, observation, *, interrupted=False, task_key="drawer_bowl", episode_index=0, override_path=None,
                 bddl_path=None, original_bddl_path=None):
    source = write_bundle(tmp_path, demo_bundle(task_key))
    reference = load_demonstration(source, task_key=task_key)
    task = resolve_distance_task(task_key)
    obs = replace(observation, prompt=task.prompt)
    base, instances = fake_simulator(obs, success_at=None if interrupted else 11)
    run = tmp_path / "demo_run"

    class DemoSimulator(base):
        def __init__(self, config):
            # The immutable reference must be durable before an episode exists.
            assert (run / "demonstration.json").read_bytes() == source.read_bytes()
            assert (run / "manifest.json").is_file()
            super().__init__(config)
            mock_override_metadata(self)

        def list_tasks(self):
            return [{"task_id": task.task_id, "prompt": task.prompt}]

        def reset(self, task_id, selected_episode):
            assert (task_id, selected_episode) == (task.task_id, episode_index)
            assert self.config.suite == task.suite
            initial = super().reset(0, 0)
            self.status.update(task_id=task_id, episode_index=selected_episode)
            mock_override_reset(self, original_bddl_path)
            return initial

    predictions = []
    phases = ("handle_grasp", "drawer_open", "handle_release") if task_key == "drawer_bowl" else ("approach", "grasp", "release")
    for phase in phases:
        raw = intent(direct=True, phase=phase)
        raw["target_region"] = "drawer_handle" if task_key == "drawer_bowl" else ("plate" if phase == "release" else "black_bowl")
        predictions.append(raw)
    responses = [envelope(raw, id=f"demo-audit-{i}", model="gpt-6-astra", reasoning={"effort": "medium"})
                 for i, raw in enumerate(predictions)]
    if interrupted:
        responses[1] = KeyboardInterrupt("fixture interrupted request")
    transport = FakeTransport(*responses)
    config = sim_config(suite=task.suite)
    if override_path is not None:
        config = replace(config, initial_state_override=str(override_path))
    if bddl_path is not None:
        config = replace(config, task_bddl_override=str(bddl_path))
    summary = runner.execute_episode(run, config=runner.configuration(), sim_config=config,
        token_provider=lambda: "fixture-oauth", transport=transport, sim_factory=DemoSimulator,
        recorder_factory=FakeRecorder, task_key=task_key, demonstration_path=source,
        episode_index=episode_index, **SETTINGS)
    assert summary["status"] == ("user_stop" if interrupted else "success"), summary.get("error")
    assert len(instances) == 1  # No second simulator episode to replay the demo.
    return run, reference, transport


@pytest.mark.parametrize("task_key", ["drawer_bowl", "bowl_plate"])
def test_external_demo_replays_once_per_request_outside_live_observation_history(
        tmp_path, observation, skip_fake_video, task_key):
    run, reference, transport = demo_fixture(tmp_path, observation, task_key=task_key)
    report = auditor.audit_run(run, tmp_path / "audit")
    assert report["passed"], report["issues"]
    assert report["demonstration_conditioned"] and report["demonstration_validated"]
    assert report["demonstration"] == reference.metadata()
    assert report["distance_task"] == task_key
    assert report["executed_actions"] == report["feedback_records"] == 11
    assert report["replayed_recorded_requests"] == 3
    assert report["demonstration"]["action_count"] == 4
    for index, payload in enumerate(transport.calls):
        assert payload["input"][0] == reference.message()
        assert sum(item == reference.message() for item in payload["input"]) == 1
        live_images = sum(part.get("type") == "input_image" for item in payload["input"][1:]
                          for part in item.get("content", []) if isinstance(part, dict))
        assert live_images == min(index + 1, 2) * 2
    assert report["artifact_sha256"]["demonstration.json"] == reference.metadata()["bundle_sha256"]
    assert not report["real_task_completion_verified"] and report["synthetic_test_fixture"]


@pytest.mark.parametrize("tamper", ["duplicate", "missing", "hidden_field", "future_evaluation_frame"])
def test_demo_prefix_tampering_is_rejected(tmp_path, observation, skip_fake_video, tamper):
    run, _, _ = demo_fixture(tmp_path, observation)
    request = sorted((run / "requests").glob("*.request.json"))[0]
    record = json.loads(request.read_text())
    inputs = record["payload"]["input"]
    if tamper == "duplicate":
        inputs.insert(0, copy.deepcopy(inputs[0]))
    elif tamper == "missing":
        inputs.pop(0)
    elif tamper == "hidden_field":
        header = json.loads(inputs[0]["content"][0]["text"])
        header["object_pose"] = [0., 0., 0.]
        inputs[0]["content"][0]["text"] = json.dumps(header)
    else:
        future = [r for r in rows(run / "trace.jsonl") if r["event"] == "executed"][-1]
        image = next(p for p in inputs[0]["content"] if p["type"] == "input_image")
        image["image_url"] = PolicyObservation.from_dict(future["observation"]).image
    rewrite(request, record)
    report = auditor.audit_run(run, tmp_path / "audit")
    assert not report["passed"]
    assert any("one exact immutable external demonstration" in issue for issue in report["issues"])


@pytest.mark.parametrize("tamper", ["hidden_state", "changed_action", "manifest_hash", "old_prompt_version"])
@pytest.mark.parametrize("task_key", ["drawer_bowl", "bowl_plate"])
def test_frozen_demo_integrity_and_explicit_conditioning_version_are_checked(
        tmp_path, observation, skip_fake_video, tamper, task_key):
    run, _, _ = demo_fixture(tmp_path, observation, task_key=task_key)
    if tamper in ("hidden_state", "changed_action"):
        path = run / "demonstration.json"
        bundle = json.loads(path.read_text())
        if tamper == "hidden_state":
            bundle["steps"][0]["object_state"] = [0., 0., 0.]
        else:
            bundle["steps"][0]["normalized_action"][0] = .02
        rewrite(path, bundle)
    else:
        path = run / "manifest.json"
        manifest = json.loads(path.read_text())
        if tamper == "manifest_hash":
            manifest["configuration"]["demonstration"]["bundle_sha256"] = "b" * 64
        else:
            manifest["configuration"]["policy"]["prompt_version"] = distance_prompt_version(task_key)
        manifest["identity_hash"] = digest(manifest["configuration"])
        rewrite(path, manifest)
    report = auditor.audit_run(run, tmp_path / "audit")
    assert not report["passed"]
    expected = {"hidden_state": "observable-only schema", "changed_action": "frozen demonstration hashes",
                "manifest_hash": "frozen demonstration hashes", "old_prompt_version": "distance protocol setting: prompt_version"}
    assert any(expected[tamper] in issue for issue in report["issues"]), report["issues"]


def test_bowl_manifest_cannot_label_external_demo_as_another_task(tmp_path, observation, skip_fake_video):
    run, _, _ = demo_fixture(tmp_path, observation, task_key="bowl_plate")
    path = run / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["configuration"]["demonstration"]["task_key"] = "drawer_bowl"
    manifest["identity_hash"] = digest(manifest["configuration"])
    rewrite(path, manifest)
    report = auditor.audit_run(run, tmp_path / "audit")
    assert not report["passed"]
    assert any("demonstration task matches the live task" in issue for issue in report["issues"])


@pytest.mark.parametrize("task_key", ["drawer_bowl", "bowl_plate"])
def test_complete_demo_does_not_complete_live_episode_or_interrupted_request(
        tmp_path, observation, skip_fake_video, task_key):
    run, reference, _ = demo_fixture(tmp_path, observation, interrupted=True, task_key=task_key)
    report = auditor.audit_run(run, tmp_path / "audit")
    assert report["passed"], report["issues"]
    assert report["status"] == "user_stop" and not report["evaluation_completed"]
    assert report["demonstration"]["all_steps_included"]
    assert report["executed_actions"] == 5 and reference.metadata()["action_count"] == 4
    assert report["replayed_recorded_requests"] == 2 and report["accepted_recorded_responses"] == 1
    assert report["trace_complete"] and not report["libero_success_observed"]
    assert report["usage_totals"]["unknown_usage_calls"] == 1
    assert report["live_model_calls_by_audit"] == report["simulator_actions_by_audit"] == 0
