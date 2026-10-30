"""Parent main-thread executive/fragment orchestration, without a live launcher.

Caller constructs NativeWorkerClient, obtains its actual READY, and supplies an
already existing human approval/consumed binding. This module neither creates
those records nor starts a process/model connection. begin checks the exact
manifest configuration and worker claim before lifecycle.start. Actual model
requests use NativeExecutiveClient on this parent process's main thread; tests
may supply explicitly marked offline mocks. There is no whole-attempt retry,
resume, automatic reset, home motion, or success inferred from model 'finish'.

run_next executes ONE accepted model decision. Caller owns wall-time/operator
policy and may call again while RUNNING. observe/finish produce zero actions and
a fresh synchronized observation. Model inputs contain only clean3RGB, public
robot/calibration, and typed fragment feedback; native evaluator/accounting data
never enters them. Original model Responses objects and decisions are preserved
separately before execution. Frozen client transport retains its own reviewed
bounded request retry policy; this runner never retries a failed decision.

Each acknowledged native action gets one three-view capture and an indexed video
frame record (head is the primary video view). No capture advances physics.
Frames remain artifacts for the caller's encoder; no encoder is launched here.
Terminal capture is permitted before closing. If post-action capture/measurement
fails, a previously verified NativeResult is completed exactly once before the
infrastructure stop, preserving native terminal truth. Current controller labels
remain candidate labels; real calibration/readiness must pass the worker gate.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import os
from pathlib import Path
import threading

import numpy as np

from .controller import ControllerConfig
from .official_control import validate_control_backend
from .fragment_executor import FragmentExecutionError, FragmentExecutor, MonitoringSettings
from .native_worker_client import (NativeWorkerClient, ParentCapture, VerifiedActionFollowupError,
                                  WorkerClientError, write_private_record)
from .schema import ExecutionLimits, public_json, validate_controller_capabilities, validate_decision, validate_feedback


class EpisodeRunnerError(RuntimeError):
    pass


def execution_configuration(*, controller_config: ControllerConfig, monitoring: MonitoringSettings,
                            limits: ExecutionLimits, initial_gripper_commands, controller_capabilities,
                            control_backend="legacy_joint") -> dict:
    control_backend = validate_control_backend(control_backend)
    controller_config.validate()
    if not isinstance(limits, ExecutionLimits): raise TypeError("ExecutionLimits required")
    capability = validate_controller_capabilities(controller_capabilities)
    c = controller_config
    if control_backend == "official_ee":
        expected = {"version": "robodojo-uc71-official-ee-controller-1",
                    "status": capability.get("status"), **monitoring.checked()}
        if capability != expected or capability.get("status") not in ("candidate_unvalidated", "validated"):
            raise EpisodeRunnerError("official_controller_capabilities_do_not_match_execution_settings")
        return {"control_backend": control_backend, "monitoring": monitoring.checked(),
                "limits": asdict(limits), "initial_gripper_commands": dict(initial_gripper_commands),
                "controller_capabilities": capability,
                "native_semantics": {"target_frame": "world_link6", "inactive_arm_hold": "cartesian_pose",
                    "ik_failure": "native_previous_joint_command_and_gripper_then_observe",
                    "scene_geometry": "official_table_and_self_collision", "custom_dls": False}}
    expected = {"version": "robodojo-uc71-robot-only-controller-1", "status": capability.get("status"),
        "action_translation_m": {"transit": c.transit_translation_speed_m_s/c.control_hz,
                                 "contact": c.contact_translation_speed_m_s/c.control_hz},
        "action_rotation_rad": c.angular_speed_rad_s/c.control_hz,
        "max_joint_step_rad": c.joint_target_lead_rad, "joint_target_lead_rad": c.joint_target_lead_rad,
        **monitoring.checked()}
    if capability != expected or capability.get("status") not in ("candidate_unvalidated", "validated"):
        raise EpisodeRunnerError("controller_capabilities_do_not_match_execution_settings")
    return {"controller": asdict(controller_config), "monitoring": monitoring.checked(),
            "limits": asdict(limits), "initial_gripper_commands": dict(initial_gripper_commands),
            "controller_capabilities": capability}


class NativeEpisodeRunner:
    def __init__(self, worker: NativeWorkerClient, *, model_client, monitoring: MonitoringSettings,
                 controller_config: ControllerConfig, limits: ExecutionLimits,
                 stop_requested=None):
        if not isinstance(worker, NativeWorkerClient): raise TypeError("NativeWorkerClient required")
        if getattr(model_client, "offline_mock", False) is not True:
            from .client import NativeExecutiveClient
            if not isinstance(model_client, NativeExecutiveClient):
                raise TypeError("Frozen native executive client or explicit offline mock required")
            log_dir = getattr(model_client, "log_dir", None)
            if log_dir is None or not Path(log_dir).resolve().is_relative_to(worker.output):
                raise EpisodeRunnerError("model_request_logs_require_parent_private_directory")
        if not callable(getattr(model_client, "request_agent", None)) or model_client.limits != limits:
            raise EpisodeRunnerError("executive_execution_contract_mismatch")
        expected = execution_configuration(controller_config=controller_config, monitoring=monitoring,
            limits=limits, initial_gripper_commands=worker.initial_gripper_commands,
            controller_capabilities=model_client.controller_capabilities,
            control_backend=worker.manifest["configuration"].get("control_backend", "legacy_joint"))
        if worker.manifest["configuration"].get("execution") != expected:
            raise EpisodeRunnerError("execution_configuration_not_in_approved_manifest")
        if stop_requested is not None and not callable(stop_requested): raise TypeError("stop callback required")
        self.worker, self.lifecycle, self.model = worker, worker.lifecycle, model_client
        self._pid, self._monitoring, self._controller, self._limits = os.getpid(), monitoring, controller_config, limits
        self._stop_requested = stop_requested or (lambda: False)
        self._control_backend = worker.manifest["configuration"].get("control_backend", "legacy_joint")
        self._executor = self._current_capture = self._last_action_capture = None
        self._feedback = self._execution = None
        self._step = 0
        self._busy = self._begun = self._closed = self._terminal_saved = False
        self.output = worker.output
        self._video = self.output / "video_frames"
        self._video.mkdir(mode=0o700, exist_ok=False)

    def _main_thread(self):
        if os.getpid() != self._pid or threading.current_thread() is not threading.main_thread():
            raise EpisodeRunnerError("model_requests_require_parent_main_thread")

    def _stopped(self):
        value = self._stop_requested()
        if type(value) is not bool: raise EpisodeRunnerError("invalid_operator_stop_state")
        return value

    def _persist_terminal(self):
        if self.lifecycle.state == "TERMINAL" and not self._terminal_saved:
            self.lifecycle.persist_terminal(self.output / "terminal.json")
            self._terminal_saved = True

    def _infrastructure_stop(self):
        self.lifecycle.stop("infrastructure_error")
        try:
            self._persist_terminal()
        finally:
            self.worker.abort()
            self._closed = True

    def _video_frame(self, index: int, capture: ParentCapture):
        write_private_record(self._video / f"frame_{index:04d}.json", {
            "schema": "robodojo-parent-video-frame-1", "scope": "operator_evidence",
            "native_action_index": index, "primary_view": "head", "images": capture.image_evidence,
            "capture_is_readonly": True, "task_success_assessed": False})

    def begin(self, existing_consumed_binding) -> None:
        self._main_thread()
        if self._begun or self._closed: raise EpisodeRunnerError("episode_begin_is_single_use")
        self._begun = True
        try:
            self.worker.begin(existing_consumed_binding)
            self._current_capture = self.worker.capture()
            self._video_frame(0, self._current_capture)
            self._executor = FragmentExecutor(lifecycle=self.lifecycle, measure=self.worker.measure,
                send_native_joint_action=self._dispatch, gripper_commands=self.worker.initial_gripper_commands,
                monitoring=self._monitoring, controller_config=self._controller, limits=self._limits,
                stop_requested=self._stopped, control_backend=self._control_backend)
        except BaseException:
            self._infrastructure_stop()
            raise

    def _dispatch(self, ticket, action):
        result = None
        try:
            receipt = self.worker.dispatch(ticket, action)
            result = receipt.native_result
            capture = self.worker.capture()  # Includes final action, even if worker is TERMINAL.
            for arm, state in receipt.measurement.arms.items():
                measured = capture.measurement.arms[arm]
                for key in vars(state):
                    if not np.allclose(getattr(state,key), getattr(measured,key), atol=1e-9, rtol=0):
                        raise WorkerClientError("post_action_capture_robot_state_changed")
                if not np.allclose(receipt.measurement.finger_position_m[arm],
                                   capture.measurement.finger_position_m[arm], atol=1e-9, rtol=0):
                    raise WorkerClientError("post_action_capture_fingers_changed")
            self._video_frame(ticket, capture)
            self._last_action_capture = capture
            return receipt  # FragmentExecutor owns the normal single completion.
        except BaseException as error:
            if isinstance(error, VerifiedActionFollowupError): result = error.native_result
            if result is not None:
                # No ActionReceipt reaches FragmentExecutor on this path.
                self.lifecycle.complete_action(ticket, result)
            self._infrastructure_stop()
            raise FragmentExecutionError("post_action_evidence_unavailable") from None

    def run_next(self) -> dict:
        self._main_thread()
        if (not self._begun or self._closed or self._busy or self._executor is None
                or self.lifecycle.state != "RUNNING"):
            raise EpisodeRunnerError("one_running_episode_decision_required")
        self._busy = True
        try:
            if self._stopped():
                self.close()
                return {"state": "TERMINAL", "model_decision_performed": False}
            self.worker.check_binding()
            directory = self.output / f"model_turn_{self._step:06d}"
            directory.mkdir(mode=0o700, exist_ok=False)
            capture = self._current_capture
            observation = capture.observation(self._step)
            write_private_record(directory / "public_observation.json", public_json({
                "step": self._step, "robot": observation["robot"], "calibration": observation["calibration"],
                "images": capture.image_evidence, "feedback": self._feedback, "execution": self._execution}))
            old_response = getattr(self.model, "last_response", None)
            try:
                answer = self.model.request_agent(observation, feedback=deepcopy(self._feedback),
                                                 execution=deepcopy(self._execution))
            except BaseException as error:
                response = getattr(self.model, "last_response", None)
                if isinstance(response, dict) and response is not old_response:
                    write_private_record(directory / "original_model_response.json", response)
                write_private_record(directory / "model_error.json", {"error_type": type(error).__name__,
                    "automatic_episode_retry": False, "actions_from_this_response": 0})
                raise
            response = getattr(self.model, "last_response", None)
            if not isinstance(response, dict) or response is old_response:
                raise EpisodeRunnerError("original_model_response_evidence_required")
            write_private_record(directory / "original_model_response.json", response)
            decision = validate_decision(answer, self._limits)
            write_private_record(directory / "accepted_decision.json", decision)
            # The plan/claim is rechecked after model latency, before any action.
            self.worker.check_binding()
            if self._stopped():
                self.close()
                return {"state": "TERMINAL", "model_decision_performed": True, "execution_performed": False}
            self._last_action_capture = None
            outcome = self._executor.run(decision)
            self._feedback = validate_feedback(outcome.feedback)
            self._execution = {"decision": decision, "progress": deepcopy(self._feedback)}
            write_private_record(directory / "fragment_outcome.json", asdict(outcome))
            self._step += 1
            if self.lifecycle.state == "TERMINAL":
                self._persist_terminal()
                self.close()
            else:
                # Zero-action decisions always yield a fresh observation. A
                # fragment's final per-action video capture is already current.
                self._current_capture = self._last_action_capture or self.worker.capture()
            return {"state": self.lifecycle.state, "model_decision_performed": True,
                    "feedback": deepcopy(self._feedback)}
        except BaseException:
            self._infrastructure_stop()
            raise
        finally:
            self._busy = False

    def close(self) -> dict:
        self._main_thread()
        if self._closed:
            return self.lifecycle.terminal_record()
        self.lifecycle.stop("user_stop")  # Existing native terminal truth is immutable.
        self._persist_terminal()
        try:
            self.worker.close()
        except BaseException:
            self.lifecycle.record_cleanup_failure("worker_close_failed")
            write_private_record(self.output / "cleanup_failure.json", {"code": "worker_close_failed"})
            self.worker.abort()
        finally:
            self._closed = True
        return self.lifecycle.terminal_record()
