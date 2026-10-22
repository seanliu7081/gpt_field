"""Offline contact audits preserve source chronology, provenance and causality."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from gpt_field.observation import PolicyObservation, image_to_data_url


spec = importlib.util.spec_from_file_location("contact_region_audit",
    Path(__file__).resolve().parents[1] / "scripts/audit_contact_region_feedback.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def observation(x=0, prompt="align a visible object"):
    image = image_to_data_url(np.full((16, 16, 3), 128, dtype=np.uint8), size=16, rotate=False)
    return PolicyObservation(image, image, (x, 0, .5, 0, 0, 0, .04, -.04), prompt)


def records(*, count=3):
    rows = [{"event": "reset", "observation": observation().to_dict(),
             "evaluator_hidden": {"object_pose": "must-not-enter-feedback"}}]
    for index in range(count):
        rows.append({"event": "action", "step": index,
                     "diagnostics": {"hidden_or_future": "ignored"}})
        rows.append({"event": "executed", "step": index,
                     "action": [.1, 0, 0, 0, 0, 0, -1],
                     "observation": observation(.001 * (index + 1)).to_dict(),
                     "diagnostic": {"object_truth": "ignored"}})
    return rows


def trace(tmp_path, rows):
    path = tmp_path / "trace.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def test_trace_has_causal_observations_and_bound(tmp_path):
    initial, endpoints = audit.load_window(trace(tmp_path, records()), maximum_actions=2)
    assert initial == observation()
    assert [obs.state[0] for obs, _ in endpoints] == [.001, .002]
    assert all(isinstance(obs, PolicyObservation) for obs, _ in endpoints)
    assert audit.load_window(trace(tmp_path, records()), maximum_actions=0)[1] == []


@pytest.mark.parametrize("change, message", [
    (lambda rows: rows[2].update(step=7), "noncontiguous"),
    (lambda rows: rows[2]["observation"].update(prompt="different task"), "task changed"),
    (lambda rows: rows.insert(2, rows[0]), "multiple scenes"),
    (lambda rows: rows[2]["observation"].update(object_pose=[0, 0, 0]), "allowlisted"),
    (lambda rows: rows[2].update(action=[0] * 6), "seven finite"),
])
def test_trace_rejects_scene_or_schema_changes(tmp_path, change, message):
    rows = records()
    change(rows)
    with pytest.raises(ValueError, match=message):
        audit.load_window(trace(tmp_path, rows))


def test_truncated_trace_is_reported(tmp_path):
    path = trace(tmp_path, records(count=1))
    with path.open("a") as file:
        file.write('{"event":')
    with pytest.raises(ValueError, match="malformed trace"):
        audit.load_window(path)


class FeedbackSpy:
    instances = []

    def __init__(self):
        self.calls = []
        self.active = False
        self.instances.append(self)

    def configure(self, obs, reference, **kwargs):
        self.calls.append(("configure", obs, reference, kwargs))
        self.initial = obs
        self.active = reference is not None
        return {"status": "seeded" if self.active else "unavailable"}

    def corrected_goal(self, obs, goal):
        self.calls.append(("goal", obs, tuple(goal)))
        assert tuple(goal) == self.initial.state[:3], "nominal must remain the causal checkpoint pose"
        return (np.asarray(goal) + [.001, 0, 0] if self.active else None,
                {"status": "proposal" if self.active else "unavailable"})

    def begin_nominal_goal(self, nominal, *, intent_key):
        self.calls.append(("nominal", tuple(nominal), intent_key))

    def invalidate(self, reason):
        self.calls.append(("invalidate", reason))
        self.active = False

    def observe(self, obs):
        self.calls.append(("observe", obs))

    def diagnostics(self):
        return {"status": "unavailable"}

    def notify_correction_applied(self, *args):
        raise AssertionError("an offline proposal was never applied")


def test_replay_is_shadow_only_and_invalidates_closed_hand():
    endpoints = [(observation(.001), [0, 0, 0, 0, 0, 0, -1]),
                 (observation(.002), [0, 0, 0, 0, 0, 0, 1]),
                 (observation(.003), [0, 0, 0, 0, 0, 0, -1])]
    result = audit.replay_window(observation(), endpoints,
        audit.reference_fixture("handle_a"), feedback_factory=FeedbackSpy, target_id="drawer_handle")
    assert result["summary"]["shadow_proposals"] == 1
    assert result["summary"]["applied_corrections"] == 0
    assert all(not row["correction_applied"] for row in result["samples"])
    assert len([c for c in FeedbackSpy.instances[-1].calls if c[0] == "configure"]) == 1
    assert result["samples"][-1]["shadow_goal"] is None
    assert [obs.state[0] for obs, _ in endpoints] == [.001, .002, .003]


def test_null_reference_is_unavailable_not_success():
    result = audit.replay_window(observation(), [(observation(.001), [0] * 6 + [-1])],
        None, feedback_factory=FeedbackSpy, target_id="bowl")
    assert result["summary"]["reference_condition"] == "unavailable"
    assert result["summary"]["shadow_proposals"] == 0
    assert result["summary"]["statuses"] == {"unavailable": 1}


def test_fixture_keeps_occluded_view_unavailable_and_goal_semantics_explicit():
    fixture = audit.reference_fixture("handle_a")
    assert fixture["external"] is None
    wrist = fixture["wrist"]
    tool, target = np.asarray(wrist["tool_region_uv"]), np.asarray(wrist["target_region_uv"])
    expected = (tool[:2] + tool[2:] - target[:2] - target[2:]) / 2
    assert wrist["desired_offset_uv"] == expected.tolist()


def fixture_source(tmp_path):
    source = tmp_path / "source"
    selected = []
    for checkpoint in audit.CHECKPOINTS:
        path = source / "preparation" / checkpoint / "observation.json"
        audit.write_json(path, observation().to_dict())
        selected.append({"id": checkpoint, "observation_path": str(path)})
        for arm in audit.ARMS:
            directory = source / "trials" / f"{checkpoint}__{arm}"
            directory.mkdir(parents=True)
            trace(directory, records(count=2))
    audit.write_json(source / "preparation/selected.json", {"checkpoints": selected})
    return source


def test_freeze_before_replay_and_detect_source_mutation(tmp_path):
    source, output = fixture_source(tmp_path), tmp_path / "output"
    manifest = audit.freeze(source, output)
    assert manifest["policy_input_eligible"] is False
    assert manifest["fixture_selection_observed_feedback_results"] is False
    assert "NOT validated grasp target" in manifest["desired_relation"]
    _, windows = audit.validated_windows(output / "offline_fixture_manifest.json")
    assert len(windows) == 8
    with pytest.raises(FileExistsError):
        audit.freeze(source, output)
    with Path(manifest["windows"][0]["trace_path"]).open("a") as file:
        file.write("\n")
    with pytest.raises(ValueError, match="archived trace changed"):
        audit.validated_windows(output / "offline_fixture_manifest.json")


def test_reject_fixture_reselection_after_freeze(tmp_path):
    source, output = fixture_source(tmp_path), tmp_path / "output"
    audit.freeze(source, output)
    path = output / "offline_fixture_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["windows"][0]["reference"] = None
    audit.write_json(path, manifest)
    with pytest.raises(ValueError, match="manifest changed"):
        audit.validated_windows(path)
