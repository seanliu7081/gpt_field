import copy
from dataclasses import asdict
import gzip
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import zlib

import numpy as np
import pytest

from gpt_field.libero import LiberoConfig
from gpt_field.observation import image_to_data_url


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/replay_bowl_state.py"
spec = importlib.util.spec_from_file_location("replay_bowl_state", SCRIPT)
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)


def observation(step, pixel=0):
    image = image_to_data_url(np.full((2, 2, 3), pixel, dtype=np.uint8), size=2)
    return {"observation/state": [step / 100, 0, 0, 0, 0, 0, 0.04, 0.04],
            "observation/image": image, "observation/wrist_image": image, "prompt": replay.TASK}


def status(step):
    return {"task_id": 0, "episode_index": 0, "steps": step, "success": False,
            "terminated": False, "horizon_reached": False,
            "initial_state_file": "/fake/init0", "initial_state_sha256": "init-hash", "bddl_sha256": "bddl-hash"}


def trace_rows(steps=2):
    rows = [{"event": "reset", "observation": observation(0), "sim_status": status(0)}]
    for step in range(steps):
        rows.append({"step": step, "state": observation(step)["observation/state"],
                     "action": [0.1, 0, 0, 0, 0, 0, -1]})
        rows.append({"event": "executed", "step": step, "next_state": observation(step + 1)["observation/state"],
                     "observation": observation(step + 1), "sim_status": status(step + 1), "success": False})
    return rows


def write_trace(path, rows, *, closed=True, partial=b""):
    raw = b"".join((json.dumps(row) + "\n").encode() for row in rows) + partial
    if closed:
        path.write_bytes(gzip.compress(raw))
    else:
        compressor = zlib.compressobj(wbits=16 + zlib.MAX_WBITS)
        path.write_bytes(compressor.compress(raw) + compressor.flush(zlib.Z_SYNC_FLUSH))


def setup_run(root, *, rows=None):
    config = LiberoConfig(suite="libero_spatial", max_steps=None,
                          python_executable="/workspace/venvs/libero-plus/bin/python",
                          libero_root="/fake/libero")
    package = Path(replay.libero_module.__file__).parent
    identity = {"task_ids": [0], "episode_indices": [0], "reference": {"task": replay.TASK},
                "simulator": asdict(config), "software": {"source_sha256": {
                    name: replay.sha256((package / name).read_bytes())
                    for name in ("__init__.py", "libero.py", "observation.py")}}}
    manifest = {"configuration": identity, "identity_hash": replay.canonical_hash(identity)}
    original = {"metadata": {"task_order_index": 0, "seed": 7},
                "tasks": [{"task_id": 0, "prompt": replay.TASK}]}
    (root / "manifest.json").write_text(json.dumps(manifest))
    (root / "simulator.json").write_text(json.dumps(original))
    write_trace(root / replay.TRACE_NAME, rows or trace_rows())
    return original


class FakeSimulator:
    def __init__(self, config, *, state_error=0, pixel=0, metadata_error=False, status_error=False):
        assert config.max_steps is None
        assert config.python_executable == "/workspace/venvs/libero-plus/bin/python"
        self.metadata = {"task_order_index": 0, "seed": 8 if metadata_error else 7}
        self.state_error, self.pixel, self.status_error = state_error, pixel, status_error
        self.status, self.closed, self.actions = {}, False, []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def list_tasks(self):
        return [{"task_id": 0, "prompt": replay.TASK}]

    def _observation(self, step):
        data = observation(step, self.pixel)
        if step:
            data["observation/state"][0] += self.state_error
        return SimpleNamespace(state=data["observation/state"], prompt=data["prompt"], to_dict=lambda: data)

    def reset(self, task, episode):
        assert (task, episode) == (0, 0)
        self.status = status(0)
        return self._observation(0)

    def step(self, action):
        self.actions.append(action)
        step = len(self.actions)
        self.status = status(step)
        if self.status_error:
            self.status["terminated"] = True
        return self._observation(step), False

    def close(self):
        self.closed = True


def test_live_gzip_snapshot_keeps_complete_records_and_ignores_partial_line(tmp_path):
    path = tmp_path / "live.gz"
    rows = trace_rows()
    write_trace(path, rows, closed=False, partial=b'{"step":2')
    before = path.read_bytes()
    recovered, info = replay.snapshot_records(path)
    assert recovered == rows
    assert not info["gzip_complete"] and info["partial_final_line_ignored"]
    assert info["snapshot_sha256"] == replay.sha256(before)
    assert path.read_bytes() == before


def test_unexecuted_decision_is_not_replayed(tmp_path):
    rows = trace_rows()
    rows.append({"step": 2, "state": observation(2)["observation/state"], "action": [0] * 7})
    _, pairs, selection = replay.select_prefix(rows, 2)
    assert len(pairs) == 2 and selection["trailing_unexecuted_decision_ignored"]
    with pytest.raises(ValueError, match="only 2 complete"):
        replay.select_prefix(rows, 3)


@pytest.mark.parametrize("mutation", ["skipped", "duplicate", "unpaired", "state", "action", "next_state", "task", "terminated"])
def test_invalid_action_trace_is_rejected(mutation):
    rows = trace_rows()
    if mutation == "skipped":
        rows[1]["step"] = 1
    elif mutation == "duplicate":
        rows.insert(2, copy.deepcopy(rows[1]))
    elif mutation == "unpaired":
        rows.pop(1)
    elif mutation == "state":
        rows[1]["state"][0] = 99
    elif mutation == "action":
        rows[1]["action"][0] = 2
    elif mutation == "next_state":
        rows[2]["next_state"][0] = 99
    elif mutation == "task":
        rows[2]["observation"]["prompt"] = "wrong task"
    elif mutation == "terminated":
        rows[2]["sim_status"]["terminated"] = True
    with pytest.raises(ValueError):
        replay.select_prefix(rows, 2)


def test_corrupt_gzip_or_complete_json_cannot_be_ignored_as_live_tail(tmp_path):
    path = tmp_path / "trace.gz"
    write_trace(path, trace_rows())
    raw = bytearray(path.read_bytes())
    raw[-8] ^= 1
    path.write_bytes(raw)
    with pytest.raises(ValueError, match="gzip is corrupt"):
        replay.snapshot_records(path)
    path.write_bytes(gzip.compress(b'{"event":"reset"}\nnot-json\n'))
    with pytest.raises(ValueError, match="Invalid complete JSON"):
        replay.snapshot_records(path)


@pytest.mark.parametrize("change", ["identity_hash", "task", "source"])
def test_identity_and_source_checks_run_before_simulation(tmp_path, change):
    setup_run(tmp_path)
    path = tmp_path / "manifest.json"
    manifest = json.loads(path.read_text())
    if change == "identity_hash":
        manifest["identity_hash"] = "invalid"
    else:
        if change == "task":
            manifest["configuration"]["episode_indices"] = [1]
        else:
            manifest["configuration"]["software"]["source_sha256"]["libero.py"] = "different"
        manifest["identity_hash"] = replay.canonical_hash(manifest["configuration"])
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        replay.replay(tmp_path, steps=2, simulator_factory=lambda config: pytest.fail("must not start simulator"))


def test_replay_checks_all_states_images_and_leaves_original_artifacts_unchanged(tmp_path):
    setup_run(tmp_path)
    originals = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    simulators = []

    def factory(config):
        simulator = FakeSimulator(config)
        simulators.append(simulator)
        return simulator

    result = replay.replay(tmp_path, steps=2, simulator_factory=factory)
    assert result["status"] == "validated" and result["completed_actions"] == 2
    assert result["state_checks"] == 5 and result["max_state_error"] == 0
    assert result["final_images_match"] is True and result["simulator_closed"]
    assert not result["scored_evaluation"] and result["astra_requests"] == 0
    assert not result["continuation_performed"]
    assert simulators[0].closed and len(simulators[0].actions) == 2
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == originals


@pytest.mark.parametrize("kwargs,expected", [
    ({"state_error": 1e-3}, "validation_failed"),
    ({"pixel": 1}, "image_mismatch_or_unavailable"),
    ({"metadata_error": True}, "validation_failed"),
    ({"status_error": True}, "validation_failed"),
])
def test_replay_rejects_state_image_metadata_or_status_drift_and_always_closes(tmp_path, kwargs, expected):
    setup_run(tmp_path)
    simulators = []

    def factory(config):
        simulator = FakeSimulator(config, **kwargs)
        simulators.append(simulator)
        return simulator

    result = replay.replay(tmp_path, steps=2, simulator_factory=factory)
    assert result["status"] == expected
    assert result["simulator_closed"] and simulators[0].closed


def test_missing_image_is_explicitly_unavailable():
    result = replay.image_comparison({}, {})
    assert all(not item["available"] and item["match"] is None for item in result.values())
