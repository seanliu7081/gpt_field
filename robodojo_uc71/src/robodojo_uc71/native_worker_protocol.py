"""Persistent worker RPC session API, without bootstrap, simulator, or model launch.

The caller supplies an already reset env, independently tracked process identity,
frozen manifest, calibrated measurement/capture callbacks and a real readiness
verifier. Current uncalibrated candidate readers cannot satisfy this API's READY
gate. No RPC accepts readiness signatures, calibration replacements, arbitrary
environment attributes, reset, retry, resume, or task-loop instructions.

The verifier receives ready/begin/dispatch/measure/capture. For ready and begin it
must read fresh sensor and start-state structures; this module computes their
hashes, checks unchanged physics, and compares to the manifest. For other phases
it must revalidate runtime calibration/readiness; start signatures may be None.
Returning 'validated' is a caller responsibility backed by measured evidence,
not something this module or a protocol request can confer on a candidate.

Parent code creates human approval and consumes it externally. begin only checks
those existing disk records. Every physical action rechecks their exact bytes,
nonce, worker/READY/manifest binding. The worker owns an independent lifecycle,
reserves the parent's next ticket, persists evaluator evidence, and completes its
own accounting. Parent lifecycle accounting remains independent.

Only response['public'] is eligible for model observation construction. 'control'
contains typed robot-only ArmState arrays; 'evaluator' contains NativeResult and
terminal evidence and MUST NOT be merged into model input. Capture accepts only
three verified RGB PNGs written by its explicit callback into the provided fresh
directory. No unrestricted native observations are passed through.

serve_one handles one framed request; the caller owns process/event-loop cleanup.
Close/EOF/errors permanently disable actions and persist terminal truth, including
an already verified success. No approval/consumption files are created here.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass, fields
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import uuid

import numpy as np
from PIL import Image

from .approval import canonical_sha256, file_sha256, validate_ready, verify_manifest
from .controller import ArmState, measured_tcp_pose
from .fragment_executor import RobotMeasurement
from .ipc import JsonLineChannel
from .lifecycle import EpisodeLifecycle
from .native_episode import NativeEpisodeActionAdapter, validate_joint_action
from .schema import ARMS, CAMERAS, public_json, validate_calibration, validate_robot, vector


@dataclass(frozen=True)
class RuntimeReadiness:
    ready: bool
    calibration_status: str
    reset_validated: bool
    reward_predicates_registered: bool
    sensor_signature: Mapping | None = None
    start_state_signature: Mapping | None = None


@dataclass(frozen=True)
class WorkerCapture:
    images: Mapping[str, Path]
    measurement: RobotMeasurement
    calibration: Mapping


class WorkerSessionError(RuntimeError):
    """Sanitized failure code; native exceptions never cross the socket."""


def _write_new(path: Path, value: Mapping) -> None:
    raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _pairs(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("Duplicate record key")
        value[key] = item
    return value


def _read_record(path: Path) -> tuple[dict, str]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 65536:
        raise WorkerSessionError("invalid_approval_record")
    with path.open("rb") as stream:
        raw = stream.read(65537)
    if len(raw) > 65536:
        raise WorkerSessionError("invalid_approval_record")
    value = json.loads(raw, object_pairs_hook=_pairs,
                       parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite record")))
    if not isinstance(value, dict):
        raise WorkerSessionError("invalid_approval_record")
    return value, hashlib.sha256(raw).hexdigest()


class NativeWorkerSession:
    def __init__(self, channel: JsonLineChannel, *, env, manifest: Mapping,
                 worker_identity: Mapping, verify_runtime_readiness, measure_robot,
                 capture_rgb, expected_layout_path: str | Path, expected_layout_sha256: str,
                 registration_evidence_id: str, initial_gripper_commands: Mapping[str, str],
                 precreated_worker_output: bool = False):
        if not isinstance(channel, JsonLineChannel) or channel.role != "worker":
            raise WorkerSessionError("worker_channel_required")
        identity = dict(worker_identity)
        if (set(identity) != {"pid", "boot_nonce"} or identity["pid"] != os.getpid()
                or type(identity["pid"]) is not int or not isinstance(identity["boot_nonce"], str)
                or len(identity["boot_nonce"]) < 16):
            raise WorkerSessionError("actual_worker_identity_required")
        if not all(callable(fn) for fn in (verify_runtime_readiness, measure_robot, capture_rgb)):
            raise WorkerSessionError("verified_runtime_callbacks_required")
        if (set(initial_gripper_commands) != set(ARMS)
                or any(value not in ("open", "close") for value in initial_gripper_commands.values())):
            raise WorkerSessionError("initial_gripper_latches_required")
        self.channel, self.env = channel, env
        self.manifest = deepcopy(dict(manifest))
        verify_manifest(self.manifest)
        self._identity = identity
        self._verify_runtime, self._measure, self._capture = verify_runtime_readiness, measure_robot, capture_rgb
        self._layout = Path(expected_layout_path).resolve(strict=True)
        self._layout_hash, self._registration_id = expected_layout_sha256, registration_evidence_id
        if not any(entry == {"path": str(self._layout), "sha256": expected_layout_sha256}
                   for entry in self.manifest["files"].values()):
            raise WorkerSessionError("layout_not_in_approved_manifest")
        self._attempt = Path(self.manifest["attempt_dir"]).resolve(strict=True)
        self.output = self._attempt / "worker"
        if type(precreated_worker_output) is not bool:
            raise WorkerSessionError("invalid_precreated_worker_output_option")
        if precreated_worker_output:
            # The parent may own the attempt root and approval files while a
            # restricted worker owns only this explicitly precreated directory.
            metadata = self.output.lstat()
            if (not stat.S_ISDIR(metadata.st_mode) or self.output.is_symlink()
                    or metadata.st_uid != os.geteuid() or stat.S_IMODE(metadata.st_mode) != 0o700
                    or self.output.resolve(strict=True).parent != self._attempt
                    or any(self.output.iterdir())):
                raise WorkerSessionError("invalid_precreated_worker_output_directory")
        else:
            self.output.mkdir(exist_ok=False)
        self.lifecycle = EpisodeLifecycle()
        self.state = "CREATED"
        self._closed = False
        self._ready = self._claim_binding = self._adapter = None
        self._ready_file_hash = None
        self._grippers = dict(initial_gripper_commands)
        self._capture_index = 0
        self._terminal_path = None

    def _physics_snapshot(self):
        clock, steps, actions = self.env.sim.sim.current_time, self.env.sim._sim_step_counter, self.env.take_action_cnt
        if (type(clock) not in (int, float) or not math.isfinite(clock) or clock < 0
                or type(steps) is not int or steps < 0 or not isinstance(actions, list) or len(actions) != 1
                or type(actions[0]) is not int or not 0 <= actions[0] <= 1400):
            raise WorkerSessionError("invalid_native_readonly_state")
        return (float(clock), steps, tuple(actions))

    def _readonly(self, callback, *args):
        before = self._physics_snapshot()
        value = callback(*args)
        if self._physics_snapshot() != before:
            raise WorkerSessionError("readonly_callback_advanced_native_state")
        return value

    def _readiness(self, phase, *, check_start=False):
        if os.getpid() != self._identity["pid"]:
            raise WorkerSessionError("worker_process_identity_changed")
        status = self._readonly(self._verify_runtime, phase)
        if (not isinstance(status, RuntimeReadiness) or status.ready is not True
                or status.calibration_status != "validated" or status.reset_validated is not True
                or status.reward_predicates_registered is not True):
            raise WorkerSessionError("runtime_not_validated")
        if check_start:
            if self.env.take_action_cnt != [0] or not isinstance(status.sensor_signature, Mapping) or not isinstance(status.start_state_signature, Mapping):
                raise WorkerSessionError("live_start_signatures_required")
            actual = {"sensor_sha256": canonical_sha256(dict(status.sensor_signature)),
                      "start_state_sha256": canonical_sha256(dict(status.start_state_signature))}
            if any(actual[key] != self.manifest[key] for key in actual):
                raise WorkerSessionError("live_ready_signature_mismatch")
            return actual

    def _verify_claim(self):
        if self._claim_binding is None:
            raise WorkerSessionError("bound_consumption_required")
        if os.getpid() != self._identity["pid"]:
            raise WorkerSessionError("worker_process_identity_changed")
        verify_manifest(self.manifest, verify_files=False)
        approval, approval_hash = _read_record(self._attempt / "approval.json")
        consumed, consumed_hash = _read_record(self._attempt / "consumed.json")
        binding = self._claim_binding
        if (approval_hash != binding["approval_sha256"] or consumed_hash != binding["consumed_sha256"]
                or file_sha256(self.output / "ready.json") != binding["ready_file_sha256"]):
            raise WorkerSessionError("approval_consumption_or_ready_changed")
        if (approval.get("schema") != "robodojo-uc71-human-approval-1"
                or approval.get("approval_mode") != "manual_chat_review"
                or approval.get("manual_browser_review") is not False
                or approval.get("manifest_sha256") != self.manifest["manifest_sha256"]
                or approval.get("preview_sha256") != self.manifest["preview_sha256"]
                or approval.get("attempt_dir") != str(self._attempt)
                or approval.get("attempts") != 1 or type(approval.get("attempts")) is not int
                or not isinstance(approval.get("user_message"), str) or not approval["user_message"].strip()):
            raise WorkerSessionError("human_approval_binding_invalid")
        if (consumed.get("schema") != "robodojo-uc71-consumed-1"
                or consumed.get("manifest_sha256") != self.manifest["manifest_sha256"]
                or consumed.get("approval_sha256") != approval_hash
                or consumed.get("worker") != self._identity
                or consumed.get("ready_sha256") != canonical_sha256(self._ready)
                or consumed.get("ready_sha256") != binding["ready_sha256"]
                or consumed.get("claim_nonce") != binding["claim_nonce"]
                or consumed.get("attempts") != 1 or type(consumed.get("attempts")) is not int):
            raise WorkerSessionError("consumed_attempt_binding_invalid")
        if str(uuid.UUID(consumed["claim_nonce"])) != consumed["claim_nonce"]:
            raise WorkerSessionError("invalid_claim_nonce")
        validate_ready(self.manifest, self._ready, expected_worker=self._identity)
        return consumed

    def _authorize_action(self, ticket, _action):
        if self.state != "RUNNING" or self.lifecycle.snapshot()["in_flight"] != ticket:
            return False
        self._verify_claim()
        self._readiness("dispatch")
        self._verify_claim()
        return True

    def _measurement(self, sample):
        if (not isinstance(sample, RobotMeasurement) or set(sample.arms) != set(ARMS)
                or set(sample.finger_position_m) != set(ARMS)):
            raise WorkerSessionError("typed_robot_measurement_required")
        control, robot, fingers = {}, {}, {}
        for arm in ARMS:
            state = sample.arms[arm]
            position, quat = measured_tcp_pose(state)
            control[arm] = {field.name: np.asarray(getattr(state, field.name)).tolist() for field in fields(ArmState)}
            fingers[arm] = vector(sample.finger_position_m[arm], 2, "measured fingers")
            robot[arm] = {"tool_pose_wxyz": [*position.tolist(), *quat.tolist()],
                "joint_position_rad": control[arm]["joint_positions"],
                "joint_velocity_rad_s": control[arm]["joint_velocities"],
                "finger_position_m": fingers[arm], "finger_gap_m": sum(fingers[arm]),
                "gripper_command": self._grippers[arm]}
        return {"control": {"arms": control, "finger_position_m": fingers},
                "public": public_json({"robot": validate_robot(robot)})}

    def _persist_terminal(self):
        if self.lifecycle.state == "TERMINAL" and self._terminal_path is None:
            self.lifecycle.persist_terminal(self.output / "terminal.json")
            self._terminal_path = self.output / "terminal.json"
        return None if self._terminal_path is None else {
            "path": str(self._terminal_path), "sha256": file_sha256(self._terminal_path),
            "record": self.lifecycle.terminal_record()}

    def _stop(self, reason):
        self._closed = True
        self.lifecycle.stop(reason)
        terminal = self._persist_terminal()
        self.state = "CLOSED" if reason == "user_stop" else "FAILED"
        return terminal

    def _handle(self, method, payload):
        if self._closed:
            raise WorkerSessionError("worker_session_closed")
        required = {"ready": set(), "begin": {"approval_sha256", "consumed_sha256", "claim_nonce", "ready_sha256"},
                    "measure": set(), "capture": set(), "action": {"ticket", "joint_action"}, "close": set()}
        if method not in required or not isinstance(payload, dict) or set(payload) != required[method]:
            raise WorkerSessionError("unsupported_worker_request")
        if method == "close":
            return {"evaluator": {"terminal": self._stop("user_stop")}, "state": "CLOSED"}
        if method == "ready":
            if self.state != "CREATED":
                raise WorkerSessionError("ready_is_single_use")
            verify_manifest(self.manifest)
            signatures = self._readiness("ready", check_start=True)
            ready = {**self._identity, "state": "READY", "reset_validated": True, "policy_actions": 0,
                     "manifest_sha256": self.manifest["manifest_sha256"], **signatures}
            validate_ready(self.manifest, ready, expected_worker=self._identity)
            _write_new(self.output / "ready.json", ready)
            self._ready_file_hash = file_sha256(self.output / "ready.json")
            self._ready = deepcopy(ready)
            self.lifecycle.ready()
            self.state = "READY"
            return deepcopy(ready)
        if method == "begin":
            if self.state != "READY":
                raise WorkerSessionError("begin_requires_ready")
            for key in ("approval_sha256", "consumed_sha256", "ready_sha256"):
                if not isinstance(payload[key], str) or re.fullmatch(r"[0-9a-f]{64}", payload[key]) is None:
                    raise WorkerSessionError("invalid_begin_digest")
            self._claim_binding = {**payload, "ready_file_sha256": self._ready_file_hash}
            consumed = self._verify_claim()
            verify_manifest(self.manifest)
            self._readiness("begin", check_start=True)
            self._verify_claim()
            self.lifecycle.start(consumed)
            self._adapter = NativeEpisodeActionAdapter(self.env, lifecycle=self.lifecycle,
                verify_authorization=self._authorize_action, expected_layout_path=self._layout,
                expected_layout_sha256=self._layout_hash, registration_evidence_id=self._registration_id,
                episode_evidence_id=consumed["claim_nonce"])
            self.state = "RUNNING"
            return {"state": "RUNNING", "claim_nonce": consumed["claim_nonce"]}
        if self.state not in ("READY", "RUNNING", "TERMINAL"):
            raise WorkerSessionError("worker_not_ready")
        if method == "measure":
            self._readiness("measure")
            return self._measurement(self._readonly(self._measure))
        if method == "capture":
            self._readiness("capture")
            self._capture_index += 1
            directory = self.output / f"capture_{self._capture_index:06d}"
            directory.mkdir(exist_ok=False)
            capture = self._readonly(self._capture, directory)
            if not isinstance(capture, WorkerCapture) or set(capture.images) != set(CAMERAS):
                raise WorkerSessionError("exact_three_rgb_capture_required")
            if len({Path(value).resolve(strict=True) for value in capture.images.values()}) != 3:
                raise WorkerSessionError("distinct_camera_rgb_paths_required")
            response = self._measurement(capture.measurement)
            images = {}
            for camera in CAMERAS:
                path = Path(capture.images[camera]).resolve(strict=True)
                if not path.is_relative_to(directory) or not 0 < path.stat().st_size <= 8 * 1024 * 1024:
                    raise WorkerSessionError("invalid_rgb_capture_path")
                with Image.open(path) as picture:
                    if picture.format != "PNG" or picture.mode != "RGB" or picture.size != (640, 480):
                        raise WorkerSessionError("native_rgb_capture_format_required")
                    picture.verify()
                images[camera] = {"path": str(path), "sha256": file_sha256(path), "bytes": path.stat().st_size}
            response["public"].update(images=images, calibration=validate_calibration(dict(capture.calibration)))
            response["public"] = public_json(response["public"])
            return response
        if self.state != "RUNNING":
            raise WorkerSessionError("actions_require_running_worker")
        ticket = payload["ticket"]
        if type(ticket) is not int or ticket != self.lifecycle.submitted_actions + 1:
            raise WorkerSessionError("parent_action_ticket_mismatch")
        action = validate_joint_action(payload["joint_action"])
        if self.lifecycle.reserve_action() != ticket:
            raise WorkerSessionError("worker_action_ticket_mismatch")
        receipt = self._adapter.dispatch(ticket, action)
        path = self.output / f"action_{ticket:04d}.json"
        _write_new(path, receipt.evaluator_receipt)
        self.lifecycle.complete_action(ticket, receipt.native_result)
        for arm in ARMS:
            self._grippers[arm] = "open" if action[f"{arm}_ee_joint_state"][0] == 1. else "close"
        if self.lifecycle.state == "TERMINAL":
            self.state = "TERMINAL"
        terminal = self._persist_terminal()
        response = self._measurement(self._readonly(self._measure))
        response["evaluator"] = {"native_result": asdict(receipt.native_result),
            "receipt": {"path": str(path), "sha256": file_sha256(path)}, "terminal": terminal}
        return response

    def serve_one(self, *, timeout_s=None) -> bool:
        """Serve one request. False means closed; any error closes and raises."""
        if self._closed:
            raise WorkerSessionError("worker_session_closed")
        request = None
        try:
            request = self.channel.receive_request(timeout_s=timeout_s)
            response = self._handle(request["method"], request["payload"])
            self.channel.respond(request["seq"], response)
            if self._closed:
                self.channel.close()
            return not self._closed
        except Exception as error:
            code = str(error) if isinstance(error, WorkerSessionError) else "worker_request_failed"
            try:
                self._stop("infrastructure_error")
                if request is not None and self.channel.state == "HANDLING_REQUEST":
                    self.channel.respond(request["seq"], error_code=code)
            finally:
                self.channel.close()
            raise WorkerSessionError(code) from None
        except BaseException:
            try:
                self._stop("infrastructure_error")
            finally:
                self.channel.close()
            raise
