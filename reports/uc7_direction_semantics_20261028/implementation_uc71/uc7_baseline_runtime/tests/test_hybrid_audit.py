"""Recorded hybrid contracts are audited using their own grid and chunk length."""

import copy
from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import numpy as np

from gpt_field.compact_field import COMPACT_FIELD_VERSION, expand_compact_field
from gpt_field.hybrid_subscription import ChatGPTPlanHybridClient, HYBRID_GRID
from gpt_field.observation import image_to_data_url
from gpt_field.schema import GridSpec
from test_compact_subscription import hybrid as compact_hybrid
from test_field_history_runner import FakeRecorder, fake_simulator, observation, sim_config
from test_hybrid_subscription_runner import runner


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("hybrid_audit", ROOT / "scripts/audit_hybrid_subscription.py")
auditor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(auditor)


def prediction(grid, chunk_size, direct):
    return {
        "controller": "contact_direct" if direct else "coarse_field",
        "target_region": "black_bowl", "near_target": direct,
        "routing_reason": "Synthetic visible alignment for an offline test.",
        "field": None if direct else {
            "stage": "fixture_hold", "orientation_target_xyzw": [0, 0, 0, 1],
            "gripper_command": "hold", "motion_mode": "hold",
            "values": [0] * grid.count, "blocked_indices": [], "unknown_indices": [],
        },
        "direct": {"actions": [[0, 0, 0, 0, 0, 0, 1]] * chunk_size} if direct else None,
    }


@pytest.mark.parametrize("grid,chunk_size", [(GridSpec(), 5), (HYBRID_GRID, 30), (HYBRID_GRID, 5)])
def test_auditor_checks_configured_direct_count_and_field_shape(grid, chunk_size):
    for direct in (False, True):
        valid = prediction(grid, chunk_size, direct)
        audit = auditor.Audit()
        assert auditor._wrapper(valid, grid, audit, "fixture: ", chunk_size)
        assert not audit.issues
        invalid = copy.deepcopy(valid)
        if direct:
            invalid["direct"]["actions"].pop()
        else:
            invalid["field"]["values"].pop()
        audit = auditor.Audit()
        assert not auditor._wrapper(invalid, grid, audit, "fixture: ", chunk_size)
        assert audit.issues


@pytest.mark.parametrize("grid,chunk_size,field_update_steps,planner_horizon,execution_steps,tail,direct_modes", [
    (GridSpec(), 5, 5, 3, 1, 1, [True, False, True]),
    (HYBRID_GRID, 30, 30, 3, 1, 1, [True, False, True]),
    (HYBRID_GRID, 5, 30, 3, 1, 1, [True, False, True]),
    (HYBRID_GRID, 5, 30, 10, 1, 1, [True, False, True]),
    (HYBRID_GRID, 5, 30, 10, 10, 1, [True, False, True]),
    (HYBRID_GRID, 5, 30, 10, 10, 11, [False, True, True, False]),
])
def test_recorded_contract_reconstructs_history_actions_and_feedback(
        tmp_path, observation, monkeypatch, grid, chunk_size, field_update_steps, planner_horizon,
        execution_steps, tail, direct_modes):
    # Force a legacy client only for the explicit legacy case; use the same runner
    # and manifest writer so schema, feedback and audit contracts must all agree.
    effort = None if grid == GridSpec() else "high"
    monkeypatch.setattr(runner, "ChatGPTPlanHybridClient", lambda *args, **kwargs:
        ChatGPTPlanHybridClient(*args, grid=grid, chunk_size=chunk_size,
                               field_update_steps=field_update_steps,
                               planner_horizon=planner_horizon,
                               field_execution_steps=execution_steps,
                               reasoning_effort=effort, **kwargs))
    calls = []

    def transport(*args):
        index = len(calls)
        calls.append(index)
        wrapper = prediction(grid, chunk_size, direct_modes[index])
        if planner_horizon == 10 and index == len(direct_modes) - 1 and not direct_modes[index]:
            wrapper["field"]["motion_mode"] = "move"
        return {"id": f"audit-fixture-{index}", "status": "completed", "model": "gpt-6-astra",
            "reasoning": {"effort": effort or "medium"},
            "output": [{"type": "message", "role": "assistant", "content": [
                {"type": "output_text", "text": json.dumps(wrapper)}]}],
            "usage": {"input_tokens": 1, "output_tokens": 1}}

    boundaries = [0]
    for direct in direct_modes[:-1]:
        boundaries.append(boundaries[-1] + (chunk_size if direct else field_update_steps))
    total_steps = boundaries[-1] + tail
    sim, _ = fake_simulator(observation, success_at=total_steps)
    run = tmp_path / "run"
    result = runner.execute_episode(run, config=runner.configuration(), sim_config=sim_config(),
        token_provider=lambda: "synthetic-audit-token", transport=transport,
        sim_factory=sim, recorder_factory=FakeRecorder, field_representation="dense")
    assert result["evaluation_completed"] and len(calls) == len(direct_modes)

    # Fixtures have no official initial-state provenance and no encoded video.
    # Retain real source-hash verification; only omit these two unrelated checks.
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "src").symlink_to(ROOT / "src", target_is_directory=True)
    (source_root / "scripts").symlink_to(ROOT / "scripts", target_is_directory=True)
    monkeypatch.setattr(auditor, "ROOT", source_root)
    monkeypatch.setattr(auditor, "_video_audit", lambda *args: ({"test_fixture": True}, 0))
    report = auditor.audit_run(run, tmp_path / "audit")
    assert report["passed"], report["issues"]
    assert report["requested_reasoning_effort"] == effort
    assert report["field_update_steps"] == field_update_steps
    assert report["direct_action_steps"] == chunk_size
    assert report["planner_horizon"] == planner_horizon
    assert report["field_execution_steps"] == execution_steps
    assert report["controller_switch_steps"] == [boundaries[index] for index in range(1, len(direct_modes))
                                                if direct_modes[index] != direct_modes[index - 1]]
    assert report["independently_recomputed_local_plans"] == (
        sum((field_update_steps + execution_steps - 1) // execution_steps
            for direct in direct_modes[:-1] if not direct)
        + ((tail + execution_steps - 1) // execution_steps if not direct_modes[-1] else 0))
    assert report["feedback_records"] == total_steps

    if effort is not None:
        # Tampering with the requested effort must be caught, even when all
        # observations, robot actions and feedback remain internally consistent.
        request_path = sorted((run / "requests").glob("*.request.json"))[0]
        request = json.loads(request_path.read_text())
        request["payload"]["reasoning"]["effort"] = "medium"
        request_path.write_text(json.dumps(request))
        tampered = auditor.audit_run(run, tmp_path / "tampered_audit")
        assert not tampered["passed"]
        assert any("requested reasoning effort" in issue for issue in tampered["issues"])
        request["payload"]["reasoning"]["effort"] = effort
        request_path.write_text(json.dumps(request))

    if chunk_size != field_update_steps:
        # A global division by 30 cannot reconstruct variable mode blocks.
        trace_path = run / "trace.jsonl"
        records = [json.loads(line) for line in trace_path.read_text().splitlines()]
        for record in records:
            if record.get("event") == "action" and record["step"] == boundaries[1]:
                record["diagnostics"]["chunk_index"] = 0
        trace_path.write_text("".join(json.dumps(record) + "\n" for record in records))
        tampered = auditor.audit_run(run, tmp_path / "tampered_chunk_audit")
        assert not tampered["passed"]
        assert any("chronological chunks" in issue for issue in tampered["issues"])

    if execution_steps > 1 and tail > 1:
        # Cached execution must retain its original plan and correct primitive.
        for record in records:
            if record.get("event") == "action" and record["step"] == boundaries[-1] + 1:
                record["diagnostics"]["plan_step"] = 0
        trace_path.write_text("".join(json.dumps(record) + "\n" for record in records))
        tampered = auditor.audit_run(run, tmp_path / "tampered_plan_audit")
        assert not tampered["passed"]
        assert any("committed local-plan execution cadence" in issue for issue in tampered["issues"])


def test_compact_run_audits_raw_prediction_expansion_history_and_execution(tmp_path, observation, monkeypatch):
    raw_predictions = [compact_hybrid(), compact_hybrid(True), compact_hybrid()]
    raw_predictions[0]["field"]["motion_mode"] = "hold"
    calls = []

    def transport(*args):
        index = len(calls)
        calls.append(index)
        return {"id": f"compact-audit-{index}", "status": "completed", "model": "gpt-6-astra",
            "reasoning": {"effort": "high"},
            "output": [{"type": "message", "role": "assistant", "content": [
                {"type": "output_text", "text": json.dumps(raw_predictions[index])}]}],
            "usage": {"input_tokens": 1, "output_tokens": 1}}

    sim, _ = fake_simulator(observation, success_at=36)
    run = tmp_path / "compact_run"
    result = runner.execute_episode(run, config=runner.configuration(), sim_config=sim_config(),
        token_provider=lambda: "synthetic-audit-token", transport=transport,
        sim_factory=sim, recorder_factory=FakeRecorder)
    assert result["evaluation_completed"] and len(calls) == 3
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["configuration"]["policy"]["field_representation"] == COMPACT_FIELD_VERSION

    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "src").symlink_to(ROOT / "src", target_is_directory=True)
    (source_root / "scripts").symlink_to(ROOT / "scripts", target_is_directory=True)
    monkeypatch.setattr(auditor, "ROOT", source_root)
    monkeypatch.setattr(auditor, "_video_audit", lambda *args: ({"test_fixture": True}, 0))
    report = auditor.audit_run(run, tmp_path / "audit")
    assert report["passed"], report["issues"]
    assert report["controller_switch_steps"] == [30, 35]
    assert report["independently_recomputed_local_plans"] == 4
    assert report["feedback_records"] == 36

    trace_path = run / "trace.jsonl"
    records = [json.loads(line) for line in trace_path.read_text().splitlines()]
    refreshed = [record for record in records if record.get("event") == "action"
                 and record["diagnostics"]["refreshed"]]
    assert [record["step"] for record in refreshed] == [0, 30, 35]
    for record, raw in zip(refreshed, raw_predictions):
        diagnostics = record["diagnostics"]
        assert diagnostics["compact_prediction"] == raw
        assert diagnostics["routing_is_model_prediction"] is True
        if raw["field"] is not None:
            assert diagnostics["hybrid_prediction"]["field"] == expand_compact_field(raw["field"], HYBRID_GRID)

    # The raw model output and locally reconstructed dense field are distinct
    # evidence. Corrupt both independently while keeping requests untouched.
    refreshed[0]["diagnostics"]["compact_prediction"]["field"]["goal_offset_m"][0] += .01
    refreshed[-1]["diagnostics"]["field_prediction"]["values"][0] += .25
    trace_path.write_text("".join(json.dumps(record) + "\n" for record in records))
    tampered = auditor.audit_run(run, tmp_path / "tampered_audit")
    assert not tampered["passed"]
    assert any("raw compact model output" in issue for issue in tampered["issues"])
    assert any("field matches local compact expansion" in issue for issue in tampered["issues"])


def test_improved_audit_reconstructs_grounding_early_events_and_short_paths(tmp_path, observation, monkeypatch):
    near_goal, direct, boundary = compact_hybrid(), compact_hybrid(True), compact_hybrid()
    near_goal["field"].update(goal_offset_m=[.3, 0, 0], goal_reference={
        "kind": "held_object", "object_center_offset_m": [.02, 0, 0],
        "desired_object_center_offset_m": [.038, 0, 0], "visual_anchor": None})
    boundary["field"].update(goal_offset_m=[.2, 0, 0], goal_reference=None)
    predictions, calls = [near_goal, direct, boundary, direct], []

    def transport(*args):
        index = len(calls)
        calls.append(index)
        return {"id": f"grounded-audit-{index}", "status": "completed", "model": "gpt-6-astra",
            "reasoning": {"effort": "medium"}, "output": [{"type": "message", "role": "assistant",
                "content": [{"type": "output_text", "text": json.dumps(predictions[index])}]}],
            "usage": {"input_tokens": 1, "output_tokens": 1}}

    sim, _ = fake_simulator(observation, success_at=10)
    run = tmp_path / "improved_run"
    result = runner.execute_episode(run, config=runner.configuration(), sim_config=sim_config(),
        token_provider=lambda: "synthetic-audit-token", transport=transport,
        sim_factory=sim, recorder_factory=FakeRecorder, field_improvements=True,
        reasoning_effort="medium", execution_mode="tracking", feedback_history="block_summary")
    assert result["evaluation_completed"] and len(calls) == 4
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "src").symlink_to(ROOT / "src", target_is_directory=True)
    (source_root / "scripts").symlink_to(ROOT / "scripts", target_is_directory=True)
    monkeypatch.setattr(auditor, "ROOT", source_root)
    monkeypatch.setattr(auditor, "_video_audit", lambda *args: ({"test_fixture": True}, 0))
    report = auditor.audit_run(run, tmp_path / "audit")
    assert report["passed"], report["issues"]
    assert report["field_improvements"]
    assert report["field_event_reasons"] == {"goal_reached": 1, "field_boundary_reached": 1}
    assert report["controller_switch_steps"] == [1, 6, 9]
    assert report["independently_recomputed_local_plans"] == 2
    trace_path = run / "trace.jsonl"
    original = [json.loads(line) for line in trace_path.read_text().splitlines()]

    def tamper(name, change, issue):
        records = copy.deepcopy(original)
        change(records)
        trace_path.write_text("".join(json.dumps(record) + "\n" for record in records))
        invalid = auditor.audit_run(run, tmp_path / name)
        assert not invalid["passed"]
        assert any(issue in message for message in invalid["issues"]), invalid["issues"]

    def wrong_cause(records):
        next(row for row in records if row["event"] == "field_control_event")["field_control_event"]["reason"] = "no_progress"

    tamper("wrong_event_cause", wrong_cause, "field refresh cause independently recomputed")

    def wrong_grounding(records):
        first = next(row for row in records if row["event"] == "action")
        first["diagnostics"]["effective_compact_field"]["goal_offset_m"][0] = .3

    tamper("wrong_grounding", wrong_grounding, "effective goal and rotation compensation")

    def wrong_short_segment(records):
        first = next(row for row in records if row["event"] == "action")
        first["diagnostics"]["plan"]["candidates"][0]["positions"][-1][0] += .01

    tamper("wrong_short_path", wrong_short_segment, "recorded local plan independently recomputed")
    trace_path.write_text("".join(json.dumps(record) + "\n" for record in original))


def test_improved_audit_replays_visual_event_from_recorded_rgb(tmp_path, observation, monkeypatch):
    patch = np.random.default_rng(8).integers(0, 256, (11, 11, 3), dtype=np.uint8)

    def image(x):
        pixels = np.full((96, 96, 3), 20, dtype=np.uint8)
        pixels[35:46, x-5:x+6] = patch
        return image_to_data_url(pixels, size=96, rotate=False)

    observation = replace(observation, image=image(60))
    field = compact_hybrid()
    field["field"].update(goal_offset_m=[.1, 0, 0], goal_reference={
        "kind": "end_effector", "object_center_offset_m": None,
        "desired_object_center_offset_m": None, "visual_anchor": {
            "moving_point_uv": [60/95, 40/95], "target_point_uv": [80/95, 40/95]}})
    predictions, calls = [field, compact_hybrid(True)], []

    def transport(*args):
        index = len(calls)
        calls.append(index)
        return {"id": f"visual-audit-{index}", "status": "completed", "model": "gpt-6-astra",
            "reasoning": {"effort": "medium"}, "output": [{"type": "message", "role": "assistant",
                "content": [{"type": "output_text", "text": json.dumps(predictions[index])}]}],
            "usage": {"input_tokens": 1, "output_tokens": 1}}

    base_sim, _ = fake_simulator(observation, success_at=3)

    class ImageSim(base_sim):
        def step(self, action):
            before = self.obs
            after, success = super().step(action)
            if len(self.actions) <= 2:
                state = list(before.state)
                state[0] += .003
                self.obs = replace(after, state=tuple(state), image=image(60 - 5 * len(self.actions)))
            return self.obs, success

    run = tmp_path / "visual_run"
    runner.execute_episode(run, config=runner.configuration(), sim_config=sim_config(),
        token_provider=lambda: "synthetic-audit-token", transport=transport,
        sim_factory=ImageSim, recorder_factory=FakeRecorder, field_improvements=True,
        reasoning_effort="medium", execution_mode="tracking", feedback_history="block_summary")
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "src").symlink_to(ROOT / "src", target_is_directory=True)
    (source_root / "scripts").symlink_to(ROOT / "scripts", target_is_directory=True)
    monkeypatch.setattr(auditor, "ROOT", source_root)
    monkeypatch.setattr(auditor, "_video_audit", lambda *args: ({"test_fixture": True}, 0))
    report = auditor.audit_run(run, tmp_path / "audit")
    assert report["passed"], report["issues"]
    assert report["field_event_reasons"] == {"visual_divergence": 1}
    assert report["controller_switch_steps"] == [2]

    trace_path = run / "trace.jsonl"
    records = [json.loads(line) for line in trace_path.read_text().splitlines()]
    event = next(row["field_control_event"] for row in records if row["event"] == "field_control_event")
    event["visual_check"]["current_distance_px"] += 5
    trace_path.write_text("".join(json.dumps(record) + "\n" for record in records))
    invalid = auditor.audit_run(run, tmp_path / "false_visual_evidence")
    assert not invalid["passed"]
    assert any("field refresh cause independently recomputed" in issue for issue in invalid["issues"])


@pytest.mark.parametrize("terminal_direct,initial_completion,effort,early_success", [
    (False, False, "medium", False), (True, False, "medium", False),
    (False, True, "high", False), (True, False, "medium", True),
])
def test_completion_audit_validates_terminal_call_without_action_and_detects_tampering(
        tmp_path, observation, monkeypatch, terminal_direct, initial_completion, effort, early_success):
    class CompletionRecorder(FakeRecorder):
        def model_completion(self, obs, **kwargs):
            self.events.append(("model_completion", kwargs))

    terminal = compact_hybrid(terminal_direct)
    terminal.update(target_region="plate", task_complete=True,
                    completion_reason="The bowl is released on the plate in the current images.")
    if terminal_direct:
        terminal["direct"]["actions"] = [[0, 0, 0, 0, 0, 0, -1]] * 5
    else:
        terminal["field"].update(motion_mode="hold", gripper_command="open")
    before = [] if initial_completion else [compact_hybrid(), compact_hybrid(True)]
    for raw in before:
        raw.update(task_complete=False, completion_reason="")
        if raw["field"] is not None:
            raw["field"]["motion_mode"] = "hold"
    predictions, calls = [*before, terminal], []

    def transport(*args):
        index = len(calls)
        calls.append(index)
        assert index < len(predictions), "Must not request inference after completion"
        return {"id": f"completion-audit-{index}", "status": "completed", "model": "gpt-6-astra",
            "reasoning": {"effort": effort},
            "output": [{"type": "message", "role": "assistant", "content": [
                {"type": "output_text", "text": json.dumps(predictions[index])}]}],
            "usage": {"input_tokens": 1, "output_tokens": 1}}

    base_sim, instances = fake_simulator(observation, success_at=1 if early_success else None)

    class CompletionSim(base_sim):
        def step(self, action):
            result = super().step(action)
            if not self.config.stop_on_success:
                self.status["terminated"] = False
            return result

    run = tmp_path / "completion_run"
    result = runner.execute_episode(run, config=runner.configuration(), sim_config=sim_config(),
        token_provider=lambda: "synthetic-audit-token", transport=transport,
        sim_factory=CompletionSim, recorder_factory=None if initial_completion else CompletionRecorder,
        reasoning_effort=effort, stop_on_model_complete=True)
    expected_steps = 1 if early_success else (0 if initial_completion else 35)
    assert result["status"] == ("success" if early_success else "model_declared_complete")
    assert result["steps"] == len(instances[0].actions) == expected_steps
    assert len(calls) == (1 if early_success else len(predictions))
    assert result["evaluation_completed"] is early_success
    assert result["libero_success"] is (True if early_success else None)
    assert result["local_plan_count"] == (1 if early_success else (0 if initial_completion else 3))

    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "src").symlink_to(ROOT / "src", target_is_directory=True)
    (source_root / "scripts").symlink_to(ROOT / "scripts", target_is_directory=True)
    monkeypatch.setattr(auditor, "ROOT", source_root)
    if not initial_completion:
        monkeypatch.setattr(auditor, "_video_audit", lambda *args: ({"test_fixture": True}, 0))
    report = auditor.audit_run(run, tmp_path / "audit")
    assert report["passed"], report["issues"]
    assert report["requested_reasoning_effort"] == effort
    assert report["model_declared_complete"] is (not early_success)
    assert report["stop_on_model_complete"] and report["stop_on_libero_success"]
    assert report["terminal_model_completion_responses"] == int(not early_success)
    assert report["validated_hybrid_responses"] == len(calls)
    assert report["executed_actions"] == report["feedback_records"] == expected_steps
    assert report["controller_switch_steps"] == ([] if initial_completion or early_success else [30])
    assert report["live_inference_calls_by_audit"] == report["simulator_actions_by_audit"] == 0
    if early_success:
        return
    if not initial_completion and not terminal_direct:
        # Historical opt-in episodes used a model-only terminal rule. Their
        # recorded configuration remains auditable after the OR rule is added.
        manifest_path = run / "manifest.json"
        original_manifest = manifest_path.read_text()
        manifest = json.loads(original_manifest)
        manifest["configuration"]["simulator"]["stop_on_success"] = False
        manifest["configuration"]["stopping_rule"].remove("LIBERO success")
        manifest["identity_hash"] = auditor.digest(manifest["configuration"])
        manifest_path.write_text(json.dumps(manifest))
        old_rule = auditor.audit_run(run, tmp_path / "original_model_only_rule")
        assert old_rule["passed"], old_rule["issues"]
        assert old_rule["stop_on_libero_success"] is False
        manifest_path.write_text(original_manifest)

    # Tampering with only the model declaration must be detected even though its
    # normalized controller payload and every preceding action remain identical.
    response_path = sorted((run / "requests").glob("*.response.json"))[-1]
    original_response = response_path.read_text()
    response = json.loads(original_response)
    raw = json.loads(response["output"][0]["content"][0]["text"])
    raw["task_complete"] = False
    response["output"][0]["content"][0]["text"] = json.dumps(raw)
    response_path.write_text(json.dumps(response))
    tampered = auditor.audit_run(run, tmp_path / "tampered_completion")
    assert not tampered["passed"]
    assert any("accepted completion declaration" in issue for issue in tampered["issues"])
    response_path.write_text(original_response)

    # A terminal record must use the current acknowledged boundary; no invented
    # extra action or shift in the stopping point is allowed.
    trace_path = run / "trace.jsonl"
    original_trace = trace_path.read_text()
    records = [json.loads(line) for line in original_trace.splitlines()]
    records[-1]["step"] += 1
    trace_path.write_text("".join(json.dumps(record) + "\n" for record in records))
    shifted = auditor.audit_run(run, tmp_path / "tampered_boundary")
    assert not shifted["passed"]
    assert any("terminal decision follows all acknowledgements" in issue for issue in shifted["issues"])
    trace_path.write_text(original_trace)

    if not initial_completion:
        previous_action = next(record for record in reversed(records) if record["event"] == "action")
        records = [json.loads(line) for line in original_trace.splitlines()]
        records.append(copy.deepcopy(previous_action))
        trace_path.write_text("".join(json.dumps(record) + "\n" for record in records))
        extra = auditor.audit_run(run, tmp_path / "tampered_extra_action")
        assert not extra["passed"]
        assert any("trace action/execution/feedback ordering" in issue for issue in extra["issues"])


def test_frozen_source_archive_preserves_run_code_but_requires_matching_reconstruction(tmp_path, monkeypatch):
    workspace, archive = tmp_path / "workspace", tmp_path / "frozen_source"
    (workspace / "src/gpt_field").mkdir(parents=True)
    (workspace / "scripts").mkdir()
    archive.mkdir()
    originals = {"planner.py": "original planner", "libero.py": "original simulator"}
    hashes = {}
    for name, content in originals.items():
        (workspace / "src/gpt_field" / name).write_text(content)
        (archive / name).write_text(content)
        hashes[name] = hashlib.sha256(content.encode()).hexdigest()
    runner_name = "evaluate_bowl_hybrid_subscription.py"
    (workspace / "scripts" / runner_name).write_text("original runner")
    (archive / runner_name).write_text("original runner")
    config = {"software": {"source_sha256": hashes},
              "runner": {"path": "scripts/" + runner_name,
                         "sha256": hashlib.sha256(b"original runner").hexdigest()}}
    monkeypatch.setattr(auditor, "ROOT", workspace)

    (workspace / "scripts" / runner_name).write_text("future stopping rule")
    (workspace / "src/gpt_field/libero.py").write_text("future simulator")
    audit = auditor.Audit()
    modules, archived, changed = auditor._source_audit(config, audit, archive)
    assert not audit.issues and (modules, archived) == (2, 3)
    assert changed == ["src/gpt_field/libero.py", "scripts/" + runner_name]

    # Current reconstruction code may not silently change while using an older
    # archive to bypass the audit's own algorithm provenance.
    (workspace / "src/gpt_field/planner.py").write_text("different planner")
    audit = auditor.Audit()
    auditor._source_audit(config, audit, archive)
    assert any("current audit reconstruction source" in issue for issue in audit.issues)
    (workspace / "src/gpt_field/planner.py").write_text(originals["planner.py"])
    (archive / "libero.py").write_text("tampered archive")
    audit = auditor.Audit()
    auditor._source_audit(config, audit, archive)
    assert any("archived source matches run manifest" in issue for issue in audit.issues)


@pytest.mark.parametrize("model_first", [False, True])
def test_live_stop_override_uses_first_completion_event_without_relabeling_summary(tmp_path, model_first):
    trigger = {"event": "model_completion" if model_first else "executed",
               "step": 5 if model_first else 4, "wall_elapsed_s": 20.0, "success": not model_first}
    events = [{"event": "observation", "step": 0, "success": False}, trigger,
              {"event": "terminal", "step": trigger["step"]}]
    (tmp_path / "episode.events.jsonl").write_text("".join(json.dumps(event) + "\n" for event in events))
    record = {"user_requested_stop": "first of LIBERO success or Astra task_complete=true", "new_model_calls": 0,
              "status": "astra_completed_natively" if model_first else "interrupt_sent_on_libero_success",
              "completion_step": trigger["step"], "completion_event_wall_s": 20.0,
              "acknowledged_actions": 5, "signal": "SIGINT"}
    path = tmp_path / "stop_rule_override.json"
    path.write_text(json.dumps(record))
    summary = {"status": "model_declared_complete" if model_first else "user_stop"}
    original = copy.deepcopy(summary)
    executed = {index: {} for index in range(5)}
    audit = auditor.Audit()
    result = auditor._stop_override_audit(tmp_path, summary, executed, {} if model_first else None, audit)
    assert not audit.issues and result == record and summary == original

    executed[5] = {}
    audit = auditor.Audit()
    auditor._stop_override_audit(tmp_path, summary, executed, {} if model_first else None, audit)
    assert any("no acknowledged robot action after first completion" in issue for issue in audit.issues)
    del executed[5]
    record["completion_step"] += 1
    path.write_text(json.dumps(record))
    audit = auditor.Audit()
    auditor._stop_override_audit(tmp_path, summary, executed, {} if model_first else None, audit)
    assert any("journal uses the first recorded" in issue for issue in audit.issues)


@pytest.mark.parametrize("controller_mode,execution_mode,feedback_history", [
    ("hybrid", "tracking", "block_summary"),
    ("hybrid", "tracking", "full"),
    ("direct", "fixed", "block_summary"),
    ("direct", "fixed", "full"),
])
def test_optimized_protocol_audit_reconstructs_summaries_tracking_and_forced_direct(
        tmp_path, observation, monkeypatch, controller_mode, execution_mode, feedback_history):
    class CompletionRecorder(FakeRecorder):
        def model_completion(self, obs, **kwargs):
            self.events.append(("model_completion", kwargs))

    if controller_mode == "hybrid":
        initial, second, terminal = compact_hybrid(), compact_hybrid(True), compact_hybrid(True)
        initial["field"]["goal_offset_m"] = [.05, .02, 0.]
        for raw in (initial, second):
            raw.update(task_complete=False, completion_reason="")
        terminal.update(target_region="plate", task_complete=True,
                        completion_reason="The bowl is released on the plate and the open gripper is clear.")
        terminal["direct"]["actions"] = [[0, 0, 0, 0, 0, 0, -1]] * 5
        expected_steps = 35
    else:
        initial = {"actions": [[.1, 0, 0, 0, 0, 0, -1]] * 5,
                   "task_complete": False, "completion_reason": ""}
        second = copy.deepcopy(initial)
        terminal = {"actions": [[0, 0, 0, 0, 0, 0, -1]] * 5,
                    "task_complete": True, "completion_reason": "The released bowl rests on the plate."}
        expected_steps = 10
    predictions, calls = [initial, second, terminal], []

    def transport(*args):
        index = len(calls)
        calls.append(index)
        return {"id": f"optimized-audit-{index}", "status": "completed", "model": "gpt-6-astra",
            "reasoning": {"effort": "medium"},
            "output": [{"type": "reasoning", "encrypted_content": f"opaque-preserved-{index}"},
                       {"type": "message", "role": "assistant", "content": [
                           {"type": "output_text", "text": json.dumps(predictions[index])}]}],
            "usage": {"input_tokens": 1, "output_tokens": 1}}

    sim, instances = fake_simulator(observation)
    run = tmp_path / "optimized"
    result = runner.execute_episode(run, config=runner.configuration(), sim_config=sim_config(),
        token_provider=lambda: "synthetic-audit-token", transport=transport,
        sim_factory=sim, recorder_factory=CompletionRecorder, reasoning_effort="medium",
        stop_on_model_complete=True, controller_mode=controller_mode,
        execution_mode=execution_mode, feedback_history=feedback_history)
    assert result["status"] == "model_declared_complete", result
    assert result["steps"] == len(instances[0].actions) == expected_steps
    assert len(calls) == 3
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "src").symlink_to(ROOT / "src", target_is_directory=True)
    (source_root / "scripts").symlink_to(ROOT / "scripts", target_is_directory=True)
    monkeypatch.setattr(auditor, "ROOT", source_root)
    monkeypatch.setattr(auditor, "_video_audit", lambda *args: ({"test_fixture": True}, 0))
    report = auditor.audit_run(run, tmp_path / "audit")
    assert report["passed"], report["issues"]
    assert report["execution_mode"] == execution_mode
    assert report["feedback_history"] == feedback_history
    assert report["controller_mode"] == controller_mode
    assert report["tracking_uses_shared_reconstruction_algorithm"] is (execution_mode == "tracking")
    assert report["live_inference_calls_by_audit"] == report["simulator_actions_by_audit"] == 0

    if feedback_history == "block_summary":
        request_path = sorted((run / "requests").glob("*.request.json"))[1]
        original = request_path.read_text()
        request = json.loads(original)
        for message in request["payload"]["input"]:
            for item in message.get("content", []):
                if item.get("type") == "input_text" and '"execution_block_summary"' in item.get("text", ""):
                    data = json.loads(item["text"])
                    data["measured_translation_path_m"] += .01
                    item["text"] = json.dumps(data)
        request_path.write_text(json.dumps(request))
        tampered = auditor.audit_run(run, tmp_path / "tampered_summary")
        assert not tampered["passed"]
        assert any("entire shared history" in issue for issue in tampered["issues"])
        request_path.write_text(original)

    trace_path = run / "trace.jsonl"
    records = [json.loads(line) for line in trace_path.read_text().splitlines()]
    if execution_mode == "tracking":
        action = next(row for row in records if row.get("event") == "action" and row["step"] == 1)
        action["diagnostics"]["commanded_delta_xyz"][0] += .01
        expected_issue = "shared follower replay"
    else:
        records[-1]["diagnostics"]["routing_is_model_prediction"] = True
        expected_issue = "terminal routing retains"
    trace_path.write_text("".join(json.dumps(record) + "\n" for record in records))
    tampered = auditor.audit_run(run, tmp_path / "tampered_execution")
    assert not tampered["passed"]
    assert any(expected_issue in issue for issue in tampered["issues"])


@pytest.mark.parametrize("guidance,grasp_guidance", [(False, False), (True, False), (True, True)])
def test_direct_evaluator_only_stop_and_centered_guidance_are_audited(
        tmp_path, observation, monkeypatch, guidance, grasp_guidance):
    calls = []

    def transport(*args):
        calls.append(args)
        # Visually stationary/released commands are ordinary actions, never a
        # terminal declaration when model completion stopping is disabled.
        return {"id": f"strict-success-audit-{len(calls)}", "status": "completed", "model": "gpt-6-astra",
            "reasoning": {"effort": "medium"},
            "output": [{"type": "message", "role": "assistant", "content": [
                {"type": "output_text", "text": json.dumps({"actions": [[0, 0, 0, 0, 0, 0, -1]] * 5})}]}],
            "usage": {"input_tokens": 1, "output_tokens": 1}}

    sim, instances = fake_simulator(observation, success_at=7)
    run = tmp_path / "evaluator_only"
    result = runner.execute_episode(run, config=runner.configuration(), sim_config=sim_config(),
        token_provider=lambda: "synthetic-audit-token", transport=transport,
        sim_factory=sim, recorder_factory=FakeRecorder, reasoning_effort="medium",
        stop_on_model_complete=False, controller_mode="direct", feedback_history="block_summary",
        centered_placement_guidance=guidance, grasp_settle_guidance=grasp_guidance)
    assert result["status"] == "success"
    assert result["steps"] == len(instances[0].actions) == 7
    assert result["libero_success"] is result["evaluation_completed"] is True
    assert result["model_declared_complete"] is False and len(calls) == 2
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "src").symlink_to(ROOT / "src", target_is_directory=True)
    (source_root / "scripts").symlink_to(ROOT / "scripts", target_is_directory=True)
    monkeypatch.setattr(auditor, "ROOT", source_root)
    monkeypatch.setattr(auditor, "_video_audit", lambda *args: ({"test_fixture": True}, 0))
    report = auditor.audit_run(run, tmp_path / "audit")
    assert report["passed"], report["issues"]
    assert report["centered_placement_guidance"] is guidance
    assert report["grasp_settle_guidance"] is grasp_guidance
    assert report["stop_on_model_complete"] is False
    assert report["stop_on_libero_success"] is True
    assert report["terminal_model_completion_responses"] == 0
    request = json.loads(sorted((run / "requests").glob("*.request.json"))[0].read_text())
    assert set(request["payload"]["text"]["format"]["schema"]["properties"]) == {"actions"}
    suffix = ("-centered-placement-grasp-settle-v5" if grasp_guidance else
              "-centered-placement-v4" if guidance else "-block_summary-v3")
    assert request["prompt_version"].endswith(suffix)
    if guidance:
        # Even a self-consistently rehashed manifest must not disguise which
        # placement prompt and schema version were actually sent.
        manifest_path = run / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        flag = "grasp_settle_guidance" if grasp_guidance else "centered_placement_guidance"
        manifest["configuration"]["policy"][flag] = False
        manifest["identity_hash"] = auditor.digest(manifest["configuration"])
        manifest_path.write_text(json.dumps(manifest))
        tampered = auditor.audit_run(run, tmp_path / "tampered_guidance")
        assert not tampered["passed"]
        assert any("protocol provenance" in issue or "declared hybrid prompt" in issue for issue in tampered["issues"])


def test_interrupted_video_can_omit_only_unacknowledged_decision_suffix():
    expected = [("observation", 0), ("direct_action", 0), ("executed", 0),
                ("routing_decision", 1), ("action_chunk", 1), ("direct_action", 1), ("terminal", 1)]
    actual = expected[:3] + [expected[-1]]
    summary = {"status": "user_stop", "pending_action_index": None, "error": {"stage": "deciding"}}
    audit = auditor.Audit()
    assert auditor._video_sequence_complete_or_interrupted(expected, actual, summary, {0: {}, 1: {}}, {0: {}}, audit)
    assert audit.notes
    # Missing evidence for an acknowledged action, a non-prefix gap, an
    # execution marker, or an ordinary successful run must still fail.
    for events, changed, acknowledged in [
        (actual, summary, {0: {}, 1: {}}),
        (expected[:2] + [expected[-1]], summary, {0: {}}),
        (actual, {**summary, "pending_action_index": 1}, {0: {}}),
        (actual, {**summary, "status": "success"}, {0: {}}),
        (actual, {**summary, "error": {"stage": "executing"}}, {0: {}}),
    ]:
        assert not auditor._video_sequence_complete_or_interrupted(
            expected, events, changed, {0: {}, 1: {}}, acknowledged, auditor.Audit())


@pytest.mark.parametrize("grasp_guidance", [False, True])
def test_guided_hybrid_audit_checks_field_direct_success_and_protocol_tampering(
        tmp_path, observation, monkeypatch, grasp_guidance):
    predictions = [compact_hybrid(), compact_hybrid(True), compact_hybrid()]
    predictions[0]["field"]["goal_offset_m"] = [.05, .02, 0.]
    predictions[2]["field"]["goal_offset_m"] = [.02, -.025, 0.]
    calls = []

    def transport(*args):
        index = len(calls)
        calls.append(index)
        return {"id": f"fair-hybrid-audit-{index}", "status": "completed", "model": "gpt-6-astra",
            "reasoning": {"effort": "medium"},
            "output": [{"type": "message", "role": "assistant", "content": [
                {"type": "output_text", "text": json.dumps(predictions[index])}]}],
            "usage": {"input_tokens": 1, "output_tokens": 1}}

    sim, instances = fake_simulator(observation, success_at=37)
    run = tmp_path / "guided_hybrid"
    result = runner.execute_episode(run, config=runner.configuration(), sim_config=sim_config(),
        token_provider=lambda: "synthetic-audit-token", transport=transport,
        sim_factory=sim, recorder_factory=FakeRecorder, reasoning_effort="medium",
        stop_on_model_complete=False, controller_mode="hybrid", execution_mode="tracking",
        feedback_history="block_summary", centered_placement_guidance=True,
        grasp_settle_guidance=grasp_guidance)
    assert result["status"] == "success" and result["libero_success"] is True
    assert result["steps"] == len(instances[0].actions) == 37 and len(calls) == 3
    assert result["coarse_field_actions"] == 32 and result["contact_direct_actions"] == 5
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "src").symlink_to(ROOT / "src", target_is_directory=True)
    (source_root / "scripts").symlink_to(ROOT / "scripts", target_is_directory=True)
    monkeypatch.setattr(auditor, "ROOT", source_root)
    monkeypatch.setattr(auditor, "_video_audit", lambda *args: ({"test_fixture": True}, 0))
    report = auditor.audit_run(run, tmp_path / "audit")
    assert report["passed"], report["issues"]
    assert report["controller_mode"] == "hybrid" and report["execution_mode"] == "tracking"
    assert report["centered_placement_guidance"] is True
    assert report["grasp_settle_guidance"] is grasp_guidance
    assert report["stop_on_model_complete"] is False and report["stop_on_libero_success"] is True
    assert report["controller_switch_steps"] == [30, 35]
    assert report["independently_recomputed_local_plans"] == 4
    request_path = sorted((run / "requests").glob("*.request.json"))[0]
    original_request = request_path.read_text()
    request = json.loads(original_request)
    suffix = "centered-placement-grasp-settle-v5" if grasp_guidance else "centered-placement-v4"
    assert request["prompt_version"] == f"astra-compact-field-direct-history-hybrid-tracking-block_summary-{suffix}"
    assert "task_complete" not in request["payload"]["text"]["format"]["schema"]["properties"]

    # The actual sent guidance is checked, including when response/action
    # evidence remains untouched. A wrapper/controller provenance mismatch is
    # separately rejected without relying on the prompt hash alone.
    request["payload"]["instructions"] = request["payload"]["instructions"].replace(
        "Place the bowl centered within the plate rim.", "Place the bowl somewhere on the plate.")
    request_path.write_text(json.dumps(request))
    altered = auditor.audit_run(run, tmp_path / "tampered_guidance")
    assert not altered["passed"]
    assert any("declared hybrid prompt exact match" in issue for issue in altered["issues"])
    request_path.write_text(original_request)
    request = json.loads(original_request)
    request["mode"] = "direct"
    request_path.write_text(json.dumps(request))
    altered = auditor.audit_run(run, tmp_path / "tampered_mode")
    assert not altered["passed"]
    assert any("recorded hybrid protocol provenance" in issue for issue in altered["issues"])
    request_path.write_text(original_request)

    summary_path = run / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["stop_on_model_complete"] = True
    summary_path.write_text(json.dumps(summary))
    altered = auditor.audit_run(run, tmp_path / "tampered_stop")
    assert not altered["passed"]
    assert any("summary protocol setting agrees with manifest: stop_on_model_complete" in issue
               for issue in altered["issues"])
