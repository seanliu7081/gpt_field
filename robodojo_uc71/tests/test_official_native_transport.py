"""Official EEF action RPC over synthetic workers; no simulator/model calls."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json

import pytest

from robodojo_uc71.native_worker_client import (NativeWorkerClient,
    VerifiedActionFollowupError, WorkerClientError)
from robodojo_uc71.native_worker_protocol import WorkerSessionError
from robodojo_uc71.ipc import RemoteError
from test_native_episode import OfficialNativeEnv, ee_action
from test_native_worker_protocol import Case


class Transport:
    def __init__(self, tmp_path):
        self.case = Case(tmp_path, env_type=OfficialNativeEnv, control_backend="official_ee")
        self.client = NativeWorkerClient(self.case.parent, manifest=self.case.manifest,
            expected_worker=self.case.identity, verify_tracked_worker=lambda value: value == self.case.identity,
            initial_gripper_commands={"left": "open", "right": "open"}, control_backend="official_ee")
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.worker = self.pool.submit(self.serve)

    def serve(self):
        try:
            while self.case.session.serve_one():
                pass
        except WorkerSessionError:
            pass

    def begin(self):
        ready = self.client.ready()
        self.client.begin(self.case.claim(ready))

    def close(self):
        self.client.abort()
        self.worker.result(timeout=3)
        self.pool.shutdown()
        self.case.close_sockets()


@pytest.fixture
def transport(tmp_path):
    value = Transport(tmp_path)
    try:
        yield value
    finally:
        value.close()


@pytest.mark.parametrize("failed_arm", [None, "left", "right"])
def test_official_action_status_roundtrip_preserves_native_action_and_gripper(transport, failed_arm):
    transport.begin()
    env, client = transport.case.env, transport.client
    if failed_arm:
        env.ik_status[failed_arm] = "Fail"
        env.joint_targets[failed_arm] = [.25] * 6
    ticket = client.lifecycle.reserve_action()
    receipt = client.dispatch(ticket, ee_action())
    client.lifecycle.complete_action(ticket, receipt.native_result)
    assert receipt.ik_status == env.ik_status
    assert env.actions == [ee_action()] and len(env.ik_calls) == 2
    assert client.lifecycle.confirmed_actions == transport.case.session.lifecycle.confirmed_actions == 1
    assert env.gripper_targets == {"left": [0.], "right": [1.]}
    if failed_arm:
        assert list(receipt.measurement.arms[failed_arm].joint_positions) == [.25] * 6
    stored = json.loads((transport.case.session.output / "action_0001.json").read_text())
    assert stored["control_backend"] == "official_ee"
    assert stored["ik_status"] == receipt.ik_status
    assert "native_private_reason" not in json.dumps(stored)
    assert "ik_status" not in client.capture().observation(1)


@pytest.mark.parametrize("tamper", ["extra_status", "different_status"])
def test_malformed_ik_feedback_keeps_verified_action_without_retry(transport, monkeypatch, tamper):
    transport.begin()
    original = transport.client._request

    def request(method, payload=None):
        response = original(method, payload)
        if method == "action":
            response = deepcopy(response)
            if tamper == "extra_status":
                response["control"]["ik_status"]["collision_detail"] = "not allowed"
            else:
                response["control"]["ik_status"]["left"] = "Fail"
        return response

    monkeypatch.setattr(transport.client, "_request", request)
    ticket = transport.client.lifecycle.reserve_action()
    with pytest.raises(VerifiedActionFollowupError) as error:
        transport.client.dispatch(ticket, ee_action())
    assert error.value.native_result.action_index == ticket
    assert len(transport.case.env.actions) == 1
    assert transport.case.session.lifecycle.confirmed_actions == 1


def test_official_session_rejects_legacy_rpc_field_without_reserving_action(tmp_path):
    case = Case(tmp_path, env_type=OfficialNativeEnv, control_backend="official_ee")
    try:
        case.begin()
        with pytest.raises(RemoteError, match="unsupported_worker_request"):
            case.rpc("action", {"ticket": 1, "joint_action": ee_action()})
        assert case.session.lifecycle.submitted_actions == 0
        assert not case.env.actions
    finally:
        case.close_sockets()


def test_client_cannot_select_backend_different_from_manifest(tmp_path):
    case = Case(tmp_path, env_type=OfficialNativeEnv, control_backend="official_ee")
    try:
        with pytest.raises(WorkerClientError, match="manifest_control_backend_mismatch"):
            NativeWorkerClient(case.parent, manifest=case.manifest, expected_worker=case.identity,
                verify_tracked_worker=lambda _: True,
                initial_gripper_commands={"left": "open", "right": "open"})
        assert not (case.attempt / "parent").exists()
    finally:
        case.close_sockets()
