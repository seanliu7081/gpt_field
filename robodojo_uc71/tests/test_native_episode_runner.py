"""Main-thread parent orchestration over synthetic native worker/socket fixtures.

All approval records are explicit synthetic fixtures under pytest tmp_path.
No IsaacSim, process bootstrap, credentials or live model request is used.
"""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
import threading

import pytest

from robodojo_uc71.approval import canonical_sha256
from robodojo_uc71.controller import ControllerConfig
from robodojo_uc71.native_episode_runner import (EpisodeRunnerError, NativeEpisodeRunner,
                                               execution_configuration)
from robodojo_uc71.native_worker_client import NativeWorkerClient, WorkerClientError
from robodojo_uc71.native_episode import validate_joint_action
from robodojo_uc71.native_worker_protocol import WorkerSessionError
from robodojo_uc71.schema import ExecutionLimits, public_json
from test_fragment_executor import monitoring, decision, short_motion
from test_native_worker_protocol import Case
from test_native_episode import action


class OfflineModel:
    offline_mock = True
    def __init__(self):
        self.limits = ExecutionLimits()
        self.controller_capabilities = {"version":"robodojo-uc71-robot-only-controller-1", "status":"validated",
            "action_translation_m":{"transit":.4/25,"contact":.1/25}, "action_rotation_rad":2./25,
            "max_joint_step_rad":.05,"joint_target_lead_rad":.05, **monitoring().checked()}
        self.answers = [decision(kind="observe")]
        self.requests = []
        self.last_response = None
        self.before_return = lambda: None
    def request_agent(self, observation, *, feedback, execution):
        assert threading.current_thread() is threading.main_thread()
        self.requests.append(deepcopy((observation, feedback, execution)))
        answer = deepcopy(self.answers.pop(0))
        self.last_response = {"id": "synthetic-response-" + str(len(self.requests)),
            "output": [{"type":"reasoning", "encrypted_content":"opaque-original-synthetic"},
                {"type":"message", "content":[{"type":"output_text", "text":json.dumps(answer)}]}]}
        self.before_return()
        return answer


class Harness:
    def __init__(self, tmp_path, *, frozen_offline_client=False):
        self.case = Case(tmp_path)
        self.model = OfflineModel()
        configuration = execution_configuration(controller_config=ControllerConfig(), monitoring=monitoring(),
            limits=self.model.limits, initial_gripper_commands={"left":"open","right":"open"},
            controller_capabilities=self.model.controller_capabilities)
        self.case.manifest["configuration"]["execution"] = configuration
        self.case.manifest.pop("manifest_sha256")
        self.case.manifest["manifest_sha256"] = canonical_sha256(self.case.manifest)
        self.case.session.manifest = deepcopy(self.case.manifest)
        self.alive = True
        self.client = NativeWorkerClient(self.case.parent, manifest=self.case.manifest,
            expected_worker=self.case.identity, verify_tracked_worker=lambda identity: self.alive and identity == self.case.identity,
            initial_gripper_commands={"left":"open","right":"open"})
        if frozen_offline_client:
            from robodojo_uc71.client import NativeExecutiveClient
            from client_fixtures import TASK, OfflineTransport, plan, frame
            self.transport = OfflineTransport()
            self.model = NativeExecutiveClient(task_prompt=TASK, approved_plan=plan(), direction_frame=frame(),
                transport=self.transport, log_dir=self.client.output/"subscription_logs",
                controller_capabilities=self.model.controller_capabilities, limits=ExecutionLimits())
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.worker = self.pool.submit(self.serve)
        self.runner = NativeEpisodeRunner(self.client, model_client=self.model, monitoring=monitoring(),
            controller_config=ControllerConfig(), limits=self.model.limits)
    def serve(self):
        try:
            while self.case.session.serve_one(): pass
        except WorkerSessionError:
            pass
    def begin(self):
        ready = self.client.ready()
        binding = self.case.claim(ready)  # Synthetic temporary user approval only.
        self.runner.begin(binding)
        return binding
    def cleanup(self):
        self.client.abort(); self.worker.result(timeout=3)
        self.case.close_sockets(); self.pool.shutdown()


@pytest.fixture
def harness(tmp_path):
    value = Harness(tmp_path)
    try: yield value
    finally: value.cleanup()


def terminal(harness):
    return json.loads((harness.client.output/"terminal.json").read_text())


def test_begin_needs_existing_approval_and_consumed_binding(harness):
    harness.client.ready()
    with pytest.raises(Exception):
        harness.runner.begin({"approval_sha256":"a"*64,"consumed_sha256":"b"*64,
            "ready_sha256":"c"*64,"claim_nonce":"00000000-0000-0000-0000-000000000001"})
    assert not (harness.case.attempt/"approval.json").exists()
    assert not (harness.case.attempt/"consumed.json").exists()
    assert harness.case.env.actions == [] and harness.model.requests == []
    assert terminal(harness)["status"] == "infrastructure_error"


@pytest.mark.parametrize("kind", ["observe", "finish"])
def test_observe_and_finish_yield_fresh_rgb_without_actions_or_success(harness, kind):
    harness.model.answers = [decision(kind=kind), decision(kind="observe")]
    harness.begin()
    harness.runner.run_next(); harness.runner.run_next()
    assert harness.case.env.actions == [] and harness.case.env.reward_manager.calls == 0
    assert harness.client.lifecycle.state == "RUNNING"
    assert [row[0]["step"] for row in harness.model.requests] == [0,1]
    assert len(list(harness.case.session.output.glob("capture_*"))) == 3
    assert harness.model.requests[1][1]["status"] == "pending_verification"
    result = harness.runner.close()
    assert result["status"] == "user_stop" and result["official_success"] is False


def test_reserved_actions_single_native_callback_and_every_action_video(harness):
    harness.model.answers = [short_motion(steps=3), decision(kind="observe")]
    harness.begin(); harness.runner.run_next(); harness.runner.run_next()
    env = harness.case.env
    assert len(env.actions) == env.reward_manager.calls == env.terminal_calls == 3
    assert env.live_steps == 30 and harness.client.lifecycle.confirmed_actions == 3
    frames = sorted((harness.client.output/"video_frames").glob("*.json"))
    assert len(frames) == 4
    assert [json.loads(path.read_text())["native_action_index"] for path in frames] == [0,1,2,3]
    for observation, feedback, execution in harness.model.requests:
        assert set(observation) == {"step","images","robot","calibration"}
        assert set(observation["images"]) == {"head","left_wrist","right_wrist"}
        public_json({"robot":observation["robot"],"calibration":observation["calibration"],"feedback":feedback,"execution":execution})
        assert "native_reward" not in json.dumps({"robot":observation["robot"],"feedback":feedback,"execution":execution})
        assert all(value.shape == (480,640,3) for value in observation["images"].values())
    raw = json.loads((harness.client.output/"model_turn_000000/original_model_response.json").read_text())
    assert raw["output"][0]["encrypted_content"] == "opaque-original-synthetic"
    assert json.loads(raw["output"][1]["content"][0]["text"])["motion"]["steps"] == 3
    assert all(action["right_arm_joint_state"] == [0.]*6 for action in env.actions)
    assert harness.client.lifecycle.action_limit == 1400


def test_real_controller_numpy_targets_are_normalized_before_native_socket_dispatch(harness,monkeypatch):
    import numpy as np
    original = harness.client.dispatch; seen = []
    def inspect(ticket,raw):
        # Exercise the actual controller output, not a hand-authored action.
        assert any(isinstance(value,np.floating) for key,values in raw.items() if "_arm_" in key for value in values)
        normalized = validate_joint_action(raw)
        assert all(type(value) is float for values in normalized.values() for value in values)
        assert all(np.array_equal(normalized[key],raw[key]) for key in raw)
        seen.append(normalized)
        return original(ticket,raw)
    monkeypatch.setattr(harness.client,"dispatch",inspect)
    harness.model.answers = [short_motion(steps=3)]
    harness.begin(); harness.runner.run_next()
    assert len(seen) == len(harness.case.env.actions) == 3
    assert harness.case.env.actions == seen
    assert all(type(value) is float for row in harness.case.env.actions for values in row.values() for value in values)
    assert harness.case.env.reward_manager.calls == harness.case.env.terminal_calls == 3


def test_native_terminal_capture_precedes_close_and_alone_certifies_success(harness):
    harness.model.answers = [short_motion(steps=3)]
    harness.case.env.reward_manager.reward = 1.
    harness.begin(); result = harness.runner.run_next()
    assert result["state"] == "TERMINAL" and harness.case.env.take_action_cnt == [1]
    assert harness.case.env.reward_manager.calls == 1
    assert terminal(harness)["status"] == "success" and terminal(harness)["official_success"] is True
    assert (harness.client.output/"video_frames/frame_0001.json").is_file()
    assert harness.case.session.lifecycle.terminal_record()["official_success"] is True
    with pytest.raises(EpisodeRunnerError): harness.runner.run_next()


def test_capture_failure_after_verified_native_success_cannot_erase_truth(harness):
    harness.model.answers = [short_motion(steps=2)]
    harness.case.env.reward_manager.reward = 1.
    harness.begin()
    def failure(_): raise ValueError("synthetic rendering failure")
    harness.case.capture_override = failure
    with pytest.raises(Exception): harness.runner.run_next()
    assert terminal(harness)["status"] == "success" and terminal(harness)["confirmed_actions"] == 1
    assert harness.case.session.lifecycle.terminal_record()["official_success"] is True
    assert len(harness.case.env.actions) == harness.case.env.reward_manager.calls == 1
    with pytest.raises(EpisodeRunnerError): harness.runner.run_next()


def test_lost_robot_response_recovers_committed_evaluator_once_without_resend(harness):
    harness.model.answers = [short_motion(steps=2)]
    harness.case.env.reward_manager.reward = 1.
    harness.begin()
    original = harness.case.measure
    def measurement():
        if harness.case.env.actions: raise ValueError("synthetic missing measurement")
        return original()
    harness.case.session._measure = measurement
    with pytest.raises(Exception): harness.runner.run_next()
    assert terminal(harness)["status"] == "success" and terminal(harness)["confirmed_actions"] == 1
    saved = json.loads((harness.client.output/"action_0001.json").read_text())
    assert saved["recovered_after_rpc_failure"] is True
    assert harness.case.env.take_action_cnt == [1] and harness.case.env.reward_manager.calls == 1


def test_nonterminal_capture_failure_confirms_action_then_stops_infrastructure(harness):
    harness.model.answers = [short_motion(steps=3)]
    harness.begin()
    harness.case.capture_override = lambda _: (_ for _ in ()).throw(ValueError("synthetic capture error"))
    with pytest.raises(Exception): harness.runner.run_next()
    result = terminal(harness)
    assert result["status"] == "infrastructure_error" and result["confirmed_actions"] == 1
    assert result["unconfirmed_action"] is None and harness.case.env.take_action_cnt == [1]


def test_worker_failure_before_native_receipt_never_retries_uncertain_action(harness):
    harness.model.answers = [short_motion(steps=3)]
    harness.case.env.fail_at_step = 2
    harness.begin()
    with pytest.raises(Exception): harness.runner.run_next()
    result = terminal(harness)
    assert result["status"] == "infrastructure_error" and result["submitted_actions"] == 1
    assert result["confirmed_actions"] == 0 and result["unconfirmed_action"] == 1
    assert len(harness.case.env.actions) == 1


def test_only_parent_main_thread_may_call_model(harness):
    harness.begin()
    with ThreadPoolExecutor(max_workers=1) as pool:
        with pytest.raises(EpisodeRunnerError, match="main_thread"):
            pool.submit(harness.runner.run_next).result(timeout=3)
    assert harness.model.requests == [] and harness.case.env.actions == []


def test_claim_is_rechecked_after_model_latency_before_action(harness):
    harness.model.answers = [short_motion(steps=1)]
    harness.begin()
    def revoke():
        path = harness.case.attempt/"consumed.json"
        value = json.loads(path.read_text()); value["claim_nonce"] = "revoked"; path.write_text(json.dumps(value))
    harness.model.before_return = revoke
    with pytest.raises(WorkerClientError): harness.runner.run_next()
    assert harness.case.env.actions == [] and len(harness.model.requests) == 1
    assert (harness.client.output/"model_turn_000000/original_model_response.json").is_file()


def test_actual_tracked_worker_liveness_required_before_model(harness):
    harness.begin(); harness.alive = False
    with pytest.raises(WorkerClientError, match="not_alive"): harness.runner.run_next()
    assert harness.model.requests == [] and harness.case.env.actions == []


def test_no_native_dispatch_without_lifecycle_reservation(harness):
    harness.begin()
    with pytest.raises(WorkerClientError, match="reserved"): harness.client.dispatch(1, action())
    assert harness.case.env.actions == []


@pytest.mark.parametrize("kind", ["privileged", "public_pose", "tool_offset", "tool_rotation"])
def test_public_control_boundary_rejects_miswired_capture(harness, kind):
    original = harness.case.session._handle
    def changed(method, payload):
        response = original(method, payload)
        if method == "capture":
            if kind == "privileged": response["public"]["native_reward"] = 1
            if kind == "public_pose": response["public"]["robot"]["left"]["tool_pose_wxyz"][0] += .01
            transform = response["public"]["calibration"]["tools"]["left"]["link6_from_tool"]
            if kind == "tool_offset": transform[0][3] += .01
            if kind == "tool_rotation": transform[0][0] = transform[1][1] = -1.
        return response
    harness.case.session._handle = changed
    with pytest.raises(WorkerClientError): harness.begin()
    assert harness.model.requests == [] and harness.case.env.actions == []


def test_model_failure_preserves_original_response_and_does_not_retry(harness):
    harness.begin()
    def fail(): raise ValueError("synthetic model-response failure")
    harness.model.before_return = fail
    with pytest.raises(ValueError): harness.runner.run_next()
    assert len(harness.model.requests) == 1 and harness.case.env.actions == []
    assert (harness.client.output/"model_turn_000000/original_model_response.json").is_file()
    assert terminal(harness)["status"] == "infrastructure_error"


def test_cleanup_failure_preserves_already_verified_success(harness, monkeypatch):
    harness.case.env.reward_manager.reward = 1.
    harness.model.answers = [short_motion(steps=1)]
    harness.begin()
    monkeypatch.setattr(harness.client, "close", lambda: (_ for _ in ()).throw(ValueError("synthetic cleanup")))
    harness.runner.run_next()
    assert terminal(harness)["official_success"] is True
    assert json.loads((harness.client.output/"cleanup_failure.json").read_text())["code"] == "worker_close_failed"


def test_frozen_executive_offline_transport_gets_only_allowlisted_public_payload(tmp_path):
    from client_fixtures import current_fields
    value = Harness(tmp_path, frozen_offline_client=True)
    try:
        value.begin(); value.runner.run_next(); value.runner.run_next()
        assert len(value.transport.calls) == 2 and not value.case.env.actions
        for payload in value.transport.calls:
            fields = current_fields(payload)
            assert fields["context"]["native_evidence"] == {"available":False}
            assert not {"native_reward","partial_score","remaining_actions","evaluator","control"} & fields.keys()
            assert set(fields["robot"]) == {"left","right"}
        saved = json.loads((value.client.output/"model_turn_000001/original_model_response.json").read_text())
        assert saved == value.model.last_response
        assert len(list((value.client.output/"subscription_logs").glob("*.response.json"))) == 2
        assert value.model.history_stats()["retained_visual_sets"] == 2
    finally:
        value.cleanup()


def test_rejected_model_output_is_saved_without_automatic_repair_or_actions(harness):
    harness.model.answers = [{**decision(kind="observe"), "privileged_reward":1}]
    harness.begin()
    with pytest.raises(ValueError): harness.runner.run_next()
    assert len(harness.model.requests) == 1 and not harness.case.env.actions
    saved = json.loads((harness.client.output/"model_turn_000000/original_model_response.json").read_text())
    assert "privileged_reward" in saved["output"][1]["content"][0]["text"]


@pytest.mark.parametrize("field", ["action_rotation_rad", "max_joint_step_rad", "joint_target_lead_rad", "position_tolerance_m"])
def test_full_controller_capability_record_cannot_disagree_with_real_settings(field):
    capability = OfflineModel().controller_capabilities
    capability[field] = {"transit":.5,"contact":.5} if field == "position_tolerance_m" else .5
    with pytest.raises(EpisodeRunnerError, match="capabilities"):
        execution_configuration(controller_config=ControllerConfig(), monitoring=monitoring(),
            limits=ExecutionLimits(), initial_gripper_commands={"left":"open","right":"open"},
            controller_capabilities=capability)
