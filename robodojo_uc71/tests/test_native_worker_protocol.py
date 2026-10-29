"""Synthetic socket sessions; all approval fixtures live only in pytest tmp_path."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path
import socket

import numpy as np
from PIL import Image
import pytest

from robodojo_uc71.approval import (authorize, build_manifest, canonical_sha256,
                                    consume_authorization, file_sha256)
from robodojo_uc71.controller import ArmState
from robodojo_uc71.fragment_executor import RobotMeasurement
from robodojo_uc71.ipc import JsonLineChannel, RemoteError
from robodojo_uc71.native_worker_protocol import (NativeWorkerSession, RuntimeReadiness,
                                                  WorkerCapture, WorkerSessionError)
from robodojo_uc71.schema import public_json
from client_fixtures import calibration
from test_native_episode import NativeEnv, action


class Case:
    def __init__(self, tmp_path, *, candidate=False, initial_override=None):
        layout = tmp_path / "arx_x5/0/make_toast_0.json"
        layout.parent.mkdir(parents=True)
        layout.write_text('{"synthetic_only":true}\n')
        self.attempt = tmp_path / "attempt"
        self.attempt.mkdir()
        self.env = NativeEnv(layout)
        self.identity = {"pid": os.getpid(), "boot_nonce": "synthetic-fresh-worker-nonce"}
        self.sensor = {"synthetic_sensors": ["head", "left_wrist", "right_wrist"]}
        self.start = {"synthetic_robot_start": [0.] * 6}
        self.status = RuntimeReadiness(True, "uncalibrated_candidate" if candidate else "validated",
            True, True, self.sensor, self.start)
        self.phases = []
        self.capture_override = None
        self.measurement_override = None
        self.manifest = build_manifest(task_key="make_toast", attempt_dir=self.attempt,
            bindings={"layout": layout}, configuration={"native_limit": 1400},
            sensor_sha256=canonical_sha256(self.sensor), start_state_sha256=canonical_sha256(self.start),
            preview_sha256="c" * 64, preparation_worker={"pid": 999, "boot_nonce": "synthetic-preview-worker-nonce"})
        parent, worker = socket.socketpair()
        self.parent = JsonLineChannel(parent, role="parent", timeout_s=2)
        self.worker = JsonLineChannel(worker, role="worker", timeout_s=2)
        self.session = NativeWorkerSession(self.worker, env=self.env, manifest=self.manifest,
            worker_identity=self.identity, verify_runtime_readiness=self.readiness,
            measure_robot=self.measure, capture_rgb=self.capture, expected_layout_path=layout,
            expected_layout_sha256=file_sha256(layout), registration_evidence_id="synthetic-registration",
            initial_gripper_commands=initial_override or {"left": "open", "right": "open"})

    def readiness(self, phase):
        self.phases.append(phase)
        return self.status

    def measure(self):
        if self.measurement_override is not None:
            return self.measurement_override()
        arms = {}
        for arm in ("left", "right"):
            q = np.asarray(self.env.actions[-1][f"{arm}_arm_joint_state"] if self.env.actions else [0.] * 6)
            arms[arm] = ArmState(q, np.zeros(6), np.zeros(3), [1., 0., 0., 0.], np.eye(6),
                -np.ones(6) * 3, np.ones(6) * 3, np.ones(6) * 5, np.zeros(3))
        return RobotMeasurement(arms, {arm: [.02, .02] for arm in arms})

    def capture(self, directory):
        if self.capture_override is not None:
            return self.capture_override(directory)
        images = {}
        for name in ("head", "left_wrist", "right_wrist"):
            path = directory / f"{name}.png"
            Image.new("RGB", (640, 480), (12, 34, 56)).save(path)
            images[name] = path
        return WorkerCapture(images, self.measure(), calibration())

    def rpc(self, method, payload=None):
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self.session.serve_one, timeout_s=2)
            try:
                response = self.parent.call(method, payload)
            except RemoteError as remote:
                with pytest.raises(WorkerSessionError) as server:
                    future.result(timeout=3)
                assert str(server.value) == remote.code
                raise
            else:
                future.result(timeout=3)
                return response

    def claim(self, ready):
        # Explicit synthetic user approval fixtures; never real validation dirs.
        authorize(self.manifest, self.attempt, user_message="SYNTHETIC UNIT TEST ONLY",
            approved_manifest_sha256=self.manifest["manifest_sha256"],
            approved_preview_sha256=self.manifest["preview_sha256"])
        consumed = consume_authorization(self.manifest, self.attempt, ready, expected_worker=self.identity)
        return {"approval_sha256": file_sha256(self.attempt / "approval.json"),
                "consumed_sha256": file_sha256(self.attempt / "consumed.json"),
                "claim_nonce": consumed["claim_nonce"], "ready_sha256": canonical_sha256(ready)}

    def begin(self):
        ready = self.rpc("ready")
        payload = self.claim(ready)
        return self.rpc("begin", payload)

    def close_sockets(self):
        self.parent.close()
        self.worker.close()


@pytest.fixture
def case(tmp_path):
    value = Case(tmp_path)
    yield value
    value.close_sockets()


def test_ready_uses_verified_live_signatures_and_never_creates_approval(case):
    ready = case.rpc("ready")
    assert ready["pid"] == os.getpid()
    assert ready["sensor_sha256"] == canonical_sha256(case.sensor)
    assert ready["start_state_sha256"] == canonical_sha256(case.start)
    assert case.phases == ["ready"]
    assert not (case.attempt / "approval.json").exists()
    assert not (case.attempt / "consumed.json").exists()
    assert case.env.actions == []


def test_uncalibrated_candidates_fail_ready_without_consumption(tmp_path):
    case = Case(tmp_path, candidate=True)
    try:
        with pytest.raises(RemoteError, match="runtime_not_validated"):
            case.rpc("ready")
        assert not (case.attempt / "approval.json").exists()
        assert not (case.attempt / "consumed.json").exists()
        assert case.session.lifecycle.terminal_record()["status"] == "infrastructure_error"
        assert not case.env.actions
    finally:
        case.close_sockets()


def test_ready_rejects_rpc_provided_spoofed_signatures(case):
    with pytest.raises(RemoteError, match="unsupported_worker_request"):
        case.rpc("ready", {"sensor_sha256": case.manifest["sensor_sha256"]})
    assert not case.phases
    assert not (case.session.output / "ready.json").exists()


def test_live_signature_mismatch_is_not_replaced_with_manifest_signature(case):
    case.status = replace(case.status, start_state_signature={"changed": True})
    with pytest.raises(RemoteError, match="live_ready_signature_mismatch"):
        case.rpc("ready")
    assert not (case.session.output / "ready.json").exists()


def test_readiness_rejects_nonfinite_native_physics_state(case):
    case.env.sim.sim.current_time = float("nan")
    with pytest.raises(RemoteError, match="invalid_native_readonly_state"):
        case.rpc("ready")
    assert not case.phases


def test_begin_requires_actual_existing_human_approval_and_consumption(case):
    case.rpc("ready")
    with pytest.raises(RemoteError):
        case.rpc("begin", {"approval_sha256": "a"*64, "consumed_sha256": "b"*64,
                           "claim_nonce": "fake", "ready_sha256": "c"*64})
    assert not (case.attempt / "approval.json").exists()
    assert not (case.attempt / "consumed.json").exists()
    assert not case.env.actions


@pytest.mark.parametrize("change", ["worker", "nonce", "ready", "approval", "consumed"])
def test_begin_rejects_claim_binding_tampering(case, change):
    ready = case.rpc("ready")
    payload = case.claim(ready)
    if change == "ready":
        (case.session.output / "ready.json").write_text(json.dumps({**ready, "pid": 999}))
    elif change in ("approval", "consumed"):
        path = case.attempt / f"{change}.json"
        path.write_bytes(path.read_bytes() + b" ")
    else:
        path = case.attempt / "consumed.json"
        consumed = json.loads(path.read_text())
        if change == "worker":
            consumed["worker"]["pid"] = 999
        else:
            consumed["claim_nonce"] = "different-nonce"
        path.write_text(json.dumps(consumed))
        payload["consumed_sha256"] = file_sha256(path)
    with pytest.raises(RemoteError):
        case.rpc("begin", payload)
    assert not case.env.actions


def test_action_persists_evaluator_receipt_and_keeps_public_robot_feedback_separate(case):
    assert case.begin()["state"] == "RUNNING"
    response = case.rpc("action", {"ticket": 1, "joint_action": action()})
    assert case.session.lifecycle.confirmed_actions == 1
    assert len(case.env.actions) == 1 and case.env.live_steps == 10
    result = response["evaluator"]["native_result"]
    assert result["action_index"] == 1 and result["callback_verified"] is True
    receipt = response["evaluator"]["receipt"]
    assert file_sha256(receipt["path"]) == receipt["sha256"]
    assert public_json(response["public"]) == response["public"]
    assert response["public"]["robot"]["left"]["gripper_command"] == "close"
    assert "native_reward" not in json.dumps(response["public"])
    assert response["control"]["arms"]["left"]["joint_positions"] == action()["left_arm_joint_state"]
    assert case.phases.count("dispatch") == 1


@pytest.mark.parametrize("filename", ["approval.json", "consumed.json", "worker/ready.json"])
def test_each_action_rechecks_bound_approval_claim_and_ready_bytes(case, filename):
    case.begin()
    path = case.attempt / filename
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(RemoteError):
        case.rpc("action", {"ticket": 1, "joint_action": action()})
    assert not case.env.actions
    assert case.session.lifecycle.terminal_record()["unconfirmed_action"] == 1
    with pytest.raises(WorkerSessionError, match="worker_session_closed"):
        case.session.serve_one(timeout_s=.01)


def test_candidate_status_reintroduced_after_begin_blocks_dispatch(case):
    case.begin()
    case.status = replace(case.status, calibration_status="uncalibrated_candidate")
    with pytest.raises(RemoteError):
        case.rpc("action", {"ticket": 1, "joint_action": action()})
    assert not case.env.actions


def test_parent_ticket_cannot_skip_or_replay_native_actions(case):
    case.begin()
    case.rpc("action", {"ticket": 1, "joint_action": action()})
    with pytest.raises(RemoteError, match="parent_action_ticket_mismatch"):
        case.rpc("action", {"ticket": 1, "joint_action": action()})
    assert len(case.env.actions) == 1
    assert case.session.lifecycle.terminal_record()["confirmed_actions"] == 1


def test_capture_returns_exact_three_verified_pngs_and_public_measurements(case):
    case.rpc("ready")
    response = case.rpc("capture")
    assert set(response["public"]["images"]) == {"head", "left_wrist", "right_wrist"}
    for image in response["public"]["images"].values():
        assert file_sha256(image["path"]) == image["sha256"]
    assert public_json(response["public"]) == response["public"]
    assert case.env.live_steps == 0 and case.env.actions == []


def test_capture_cannot_pass_through_unrestricted_native_observations(case):
    case.rpc("ready")
    case.capture_override = lambda directory: {"reward": 1, "object_poses": {}}
    with pytest.raises(RemoteError, match="exact_three_rgb_capture_required"):
        case.rpc("capture")
    assert not case.env.actions


def test_capture_cannot_alias_one_png_as_all_three_cameras(case):
    case.rpc("ready")
    def aliased(directory):
        path = directory / "one.png"
        Image.new("RGB", (640, 480)).save(path)
        return WorkerCapture({name: path for name in ("head", "left_wrist", "right_wrist")},
                             case.measure(), calibration())
    case.capture_override = aliased
    with pytest.raises(RemoteError, match="distinct_camera_rgb_paths_required"):
        case.rpc("capture")


def test_readonly_capture_cannot_advance_physics(case):
    case.rpc("ready")
    original = case.capture
    def changed(directory):
        case.env.sim_step()
        case.capture_override = None
        return original(directory)
    case.capture_override = changed
    with pytest.raises(RemoteError, match="readonly_callback_advanced_native_state"):
        case.rpc("capture")


@pytest.mark.parametrize("method", ["reset", "resume", "get_env_attribute", "run_task"])
def test_unrestricted_or_retry_rpc_methods_are_absent(case, method):
    with pytest.raises(RemoteError, match="unsupported_worker_request"):
        case.rpc(method)
    assert not case.env.actions


def test_close_preserves_native_success_and_poisoned_session_cannot_act(case):
    case.begin()
    case.env.reward_manager.reward = 1.
    action_response = case.rpc("action", {"ticket": 1, "joint_action": action()})
    terminal = action_response["evaluator"]["terminal"]
    assert terminal["record"]["official_success"] is True
    closed = case.rpc("close")
    assert closed["evaluator"]["terminal"] == terminal
    assert file_sha256(terminal["path"]) == terminal["sha256"]
    with pytest.raises(WorkerSessionError, match="worker_session_closed"):
        case.session.serve_one(timeout_s=.01)


def test_eof_preserves_already_verified_terminal_truth(case):
    case.begin()
    case.env.reward_manager.reward = 1.
    response = case.rpc("action", {"ticket": 1, "joint_action": action()})
    terminal = response["evaluator"]["terminal"]
    case.parent.close()
    with pytest.raises(WorkerSessionError):
        case.session.serve_one(timeout_s=.01)
    assert case.session.lifecycle.terminal_record() == terminal["record"]
    assert file_sha256(terminal["path"]) == terminal["sha256"]


def test_measurement_failure_after_native_success_cannot_erase_persisted_success(case):
    case.begin()
    case.env.reward_manager.reward = 1.
    case.measurement_override = lambda: {"scene_state": "must not escape"}
    with pytest.raises(RemoteError, match="typed_robot_measurement_required"):
        case.rpc("action", {"ticket": 1, "joint_action": action()})
    terminal = json.loads((case.session.output / "terminal.json").read_text())
    assert terminal["official_success"] is True and terminal["confirmed_actions"] == 1
    assert (case.session.output / "action_0001.json").is_file()
