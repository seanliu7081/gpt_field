"""Parent-only client for an already tracked, reset NativeWorkerSession.

No process bootstrap, credentials, approval creation/consumption, reset or retry.
READY comes from the worker. begin requires existing bound human approval and
consumed.json; identity liveness is checked independently by the mandatory caller
callback before each request. The shared parent lifecycle always has horizon1400.

dispatch accepts an already reserved parent ticket and returns ActionReceipt;
FragmentExecutor normally completes it exactly once. If a verified evaluator
result is followed by a malformed robot response, VerifiedActionFollowupError
carries that result so the runner can complete accounting before stopping.
No evaluator data is included in observation(). Capture uses only three clean
RGB files and the explicit public robot/calibration projection. Raw Jacobians,
limits and evaluator receipts stay in parent-private operator artifacts.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, fields
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import uuid

import numpy as np
from PIL import Image

from .approval import canonical_sha256, file_sha256, validate_ready, verify_manifest
from .controller import ArmState, measured_tcp_pose
from .fragment_executor import ActionReceipt, RobotMeasurement
from .ipc import JsonLineChannel
from .lifecycle import EpisodeLifecycle, NativeResult, load_terminal_or_crash
from .native_episode import validate_joint_action
from .schema import (ARMS, CAMERAS, calibration_binding, public_json, validate_calibration,
                     validate_observation, validate_robot, vector)


class WorkerClientError(RuntimeError):
    """Sanitized parent infrastructure failure; never suitable as model feedback."""


class VerifiedActionFollowupError(WorkerClientError):
    def __init__(self, native_result: NativeResult):
        super().__init__("verified_action_followup_failed")
        self.native_result = native_result


def write_private_record(path: Path, value: Mapping) -> None:
    raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)


def _pairs(rows):
    result = {}
    for key, value in rows:
        if key in result: raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def _record(path: Path, *, maximum=65536):
    if path.is_symlink() or not path.is_file() or path.stat().st_size > maximum:
        raise WorkerClientError("invalid_bound_record")
    with path.open("rb") as stream: raw = stream.read(maximum+1)
    if len(raw) > maximum: raise WorkerClientError("invalid_bound_record")
    value = json.loads(raw, object_pairs_hook=_pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite record")))
    if not isinstance(value, dict): raise WorkerClientError("invalid_bound_record")
    return value, hashlib.sha256(raw).hexdigest()


def _keys(value, expected, reason):
    if not isinstance(value, dict) or set(value) != set(expected): raise WorkerClientError(reason)


@dataclass(frozen=True)
class ParentCapture:
    measurement: RobotMeasurement
    robot: dict
    calibration: dict
    images: dict
    image_evidence: dict

    def observation(self, step: int) -> dict:
        return validate_observation({"step": step, "images": {k: v.copy() for k,v in self.images.items()},
            "robot": deepcopy(self.robot), "calibration": deepcopy(self.calibration)})


class NativeWorkerClient:
    def __init__(self, channel: JsonLineChannel, *, manifest: Mapping, expected_worker: Mapping,
                 verify_tracked_worker, initial_gripper_commands: Mapping[str, str]):
        if not isinstance(channel, JsonLineChannel) or channel.role != "parent" or not callable(verify_tracked_worker):
            raise WorkerClientError("tracked_parent_channel_required")
        identity = dict(expected_worker)
        _keys(identity, {"pid", "boot_nonce"}, "tracked_worker_identity_required")
        if (type(identity["pid"]) is not int or identity["pid"] < 1 or not isinstance(identity["boot_nonce"], str)
                or len(identity["boot_nonce"]) < 16):
            raise WorkerClientError("tracked_worker_identity_required")
        if set(initial_gripper_commands) != set(ARMS) or any(v not in ("open", "close") for v in initial_gripper_commands.values()):
            raise WorkerClientError("initial_gripper_commands_required")
        self.manifest = deepcopy(dict(manifest)); verify_manifest(self.manifest)
        self.channel, self.identity, self._tracked = channel, identity, verify_tracked_worker
        self._pid = os.getpid()
        self._attempt = Path(self.manifest["attempt_dir"]).resolve(strict=True)
        self.worker_output = self._attempt / "worker"
        self.output = self._attempt / "parent"
        self.output.mkdir(mode=0o700, exist_ok=False)
        self.initial_gripper_commands = dict(initial_gripper_commands)
        self._grippers = dict(initial_gripper_commands)
        self.lifecycle = EpisodeLifecycle(action_limit=1400)
        self.state = "CREATED"
        self._ready = self._binding = None
        self._ready_hash = self._fixed_calibration = None
        self._last_ticket = self._capture_index = 0
        self._verified_timings = {}
        self.last_worker_terminal = None

    def _alive(self):
        if os.getpid() != self._pid or self._tracked(deepcopy(self.identity)) is not True:
            raise WorkerClientError("tracked_worker_not_alive")

    def _poison(self):
        self.state = "FAILED"
        self.channel.close()

    def _request(self, method, payload=None):
        try:
            if self.state in ("FAILED", "CLOSED"): raise WorkerClientError("parent_worker_client_closed")
            self._alive()
            return self.channel.call(method, payload)
        except BaseException:
            self._poison()
            raise

    def ready(self) -> dict:
        try:
            if self.state != "CREATED": raise WorkerClientError("ready_is_single_use")
            ready = self._request("ready")
            _keys(ready, {"pid", "boot_nonce", "state", "reset_validated", "policy_actions",
                "manifest_sha256", "sensor_sha256", "start_state_sha256"}, "invalid_ready_response")
            validate_ready(self.manifest, ready, expected_worker=self.identity)
            stored, digest = _record(self.worker_output / "ready.json")
            if stored != ready: raise WorkerClientError("worker_ready_record_differs")
            self._ready, self._ready_hash = deepcopy(ready), digest
            write_private_record(self.output / "ready.json", ready)
            self.lifecycle.ready(); self.state = "READY"
            return deepcopy(ready)
        except BaseException:
            self._poison(); raise

    def check_binding(self, *, verify_files=False) -> dict:
        self._alive()
        if self._binding is None or self._ready is None: raise WorkerClientError("existing_consumed_binding_required")
        verify_manifest(self.manifest, verify_files=verify_files)
        approval, approved_hash = _record(self._attempt / "approval.json")
        consumed, consumed_hash = _record(self._attempt / "consumed.json")
        ready, ready_hash = _record(self.worker_output / "ready.json")
        if (approved_hash != self._binding["approval_sha256"] or consumed_hash != self._binding["consumed_sha256"]
                or ready_hash != self._ready_hash or ready != self._ready):
            raise WorkerClientError("approval_consumption_or_ready_changed")
        expected = {"schema": "robodojo-uc71-human-approval-1", "approval_mode": "manual_chat_review",
            "manual_browser_review": False, "manifest_sha256": self.manifest["manifest_sha256"],
            "preview_sha256": self.manifest["preview_sha256"], "attempt_dir": str(self._attempt), "attempts": 1}
        if (any(type(approval.get(k)) is not type(v) or approval.get(k) != v for k,v in expected.items())
                or not isinstance(approval.get("user_message"), str) or not approval["user_message"].strip()):
            raise WorkerClientError("human_approval_binding_invalid")
        expected = {"schema": "robodojo-uc71-consumed-1", "manifest_sha256": self.manifest["manifest_sha256"],
            "approval_sha256": approved_hash, "worker": self.identity, "ready_sha256": canonical_sha256(ready),
            "claim_nonce": self._binding["claim_nonce"], "attempts": 1}
        if (any(type(consumed.get(k)) is not type(v) or consumed.get(k) != v for k,v in expected.items())
                or consumed["ready_sha256"] != self._binding["ready_sha256"]
                or str(uuid.UUID(consumed["claim_nonce"])) != consumed["claim_nonce"]):
            raise WorkerClientError("consumed_attempt_binding_invalid")
        validate_ready(self.manifest, ready, expected_worker=self.identity)
        return consumed

    def begin(self, existing_consumed_binding: Mapping) -> None:
        try:
            if self.state != "READY": raise WorkerClientError("begin_requires_ready")
            binding = deepcopy(dict(existing_consumed_binding))
            _keys(binding, {"approval_sha256", "consumed_sha256", "claim_nonce", "ready_sha256"}, "invalid_consumed_binding")
            for name in ("approval_sha256", "consumed_sha256", "ready_sha256"):
                if not isinstance(binding[name], str) or re.fullmatch(r"[0-9a-f]{64}", binding[name]) is None:
                    raise WorkerClientError("invalid_consumed_digest")
            self._binding = binding
            consumed = self.check_binding(verify_files=True)
            response = self._request("begin", binding)
            if response != {"state": "RUNNING", "claim_nonce": binding["claim_nonce"]}:
                raise WorkerClientError("invalid_begin_response")
            if self.check_binding() != consumed: raise WorkerClientError("consumed_binding_changed_during_begin")
            self.lifecycle.start(consumed); self.state = "RUNNING"
            write_private_record(self.output / "begin.json", {"binding": binding, "worker": self.identity})
        except BaseException:
            self._poison(); raise

    def _measurement(self, response, *, capture=False, evaluator=False):
        _keys(response, {"control", "public"} | ({"evaluator"} if evaluator else set()), "unexpected_worker_response_fields")
        control, public = response["control"], response["public"]
        _keys(control, {"arms", "finger_position_m"}, "unexpected_control_fields")
        _keys(control["arms"], ARMS, "both_control_arms_required")
        _keys(control["finger_position_m"], ARMS, "both_measured_fingers_required")
        _keys(public, {"robot", "images", "calibration"} if capture else {"robot"}, "unexpected_public_fields")
        robot = validate_robot(public_json(public)["robot"])
        arms, fingers = {}, {}
        for arm in ARMS:
            values = control["arms"][arm]
            _keys(values, {field.name for field in fields(ArmState)}, "unexpected_arm_state_fields")
            state = ArmState(**deepcopy(values)); position, quat = measured_tcp_pose(state)
            finger = vector(control["finger_position_m"][arm], 2, "measured fingers")
            comparisons = ((robot[arm]["tool_pose_wxyz"], [*position, *quat]),
                (robot[arm]["joint_position_rad"], state.joint_positions),
                (robot[arm]["joint_velocity_rad_s"], state.joint_velocities),
                (robot[arm]["finger_position_m"], finger))
            if (any(not np.allclose(a,b,atol=1e-9,rtol=0) for a,b in comparisons)
                    or robot[arm]["gripper_command"] != self._grippers[arm]):
                raise WorkerClientError("public_control_robot_disagreement")
            arms[arm], fingers[arm] = state, finger
        return RobotMeasurement(arms, fingers), robot

    def measure(self) -> RobotMeasurement:
        try:
            if self.state != "RUNNING": raise WorkerClientError("measurement_requires_running_parent")
            self.check_binding()
            return self._measurement(self._request("measure"))[0]
        except BaseException:
            self._poison(); raise

    def capture(self) -> ParentCapture:
        try:
            if self.state not in ("READY", "RUNNING", "TERMINAL"): raise WorkerClientError("capture_requires_ready_worker")
            if self._binding is not None: self.check_binding()
            self._capture_index += 1
            response = self._request("capture")
            measured, robot = self._measurement(response, capture=True)
            public = response["public"]
            calibration = validate_calibration(public["calibration"])
            for arm in ARMS:
                transform = np.asarray(calibration["tools"][arm]["link6_from_tool"])
                if (not np.allclose(transform[:3,:3], np.eye(3), atol=1e-9, rtol=0)
                        or not np.allclose(transform[:3,3], measured.arms[arm].tcp_offset_link6, atol=1e-9, rtol=0)):
                    raise WorkerClientError("public_control_tool_frame_disagreement")
            fixed = canonical_sha256(calibration_binding(calibration))
            if self._fixed_calibration is not None and self._fixed_calibration != fixed:
                raise WorkerClientError("fixed_calibration_changed")
            self._fixed_calibration = fixed
            evidence, images = public["images"], {}
            _keys(evidence, CAMERAS, "exact_three_rgb_images_required")
            directory = self.worker_output / f"capture_{self._capture_index:06d}"
            paths = set()
            for camera in CAMERAS:
                spec = evidence[camera]; _keys(spec, {"path", "sha256", "bytes"}, "invalid_image_evidence")
                path = Path(spec["path"])
                if (path.resolve(strict=True) != path or not path.is_relative_to(directory)
                        or path in paths or type(spec["bytes"]) is not int or not 0 < spec["bytes"] <= 8*1024*1024):
                    raise WorkerClientError("invalid_rgb_path")
                paths.add(path)
                with path.open("rb") as stream: raw = stream.read(8*1024*1024+1)
                if len(raw) != spec["bytes"] or hashlib.sha256(raw).hexdigest() != spec["sha256"]:
                    raise WorkerClientError("rgb_hash_or_size_changed")
                with Image.open(io.BytesIO(raw)) as image:
                    if image.format != "PNG" or image.mode != "RGB" or image.size != (640,480):
                        raise WorkerClientError("clean_native_rgb_required")
                    image.load(); images[camera] = np.asarray(image).copy()
            write_private_record(self.output / f"capture_{self._capture_index:06d}.json", {
                "schema": "robodojo-parent-capture-1", "public": public,
                "worker_readonly_capture_contract_verified": True})
            return ParentCapture(measured, robot, calibration, images, deepcopy(evidence))
        except BaseException:
            self._poison(); raise

    def _terminal(self, spec):
        if spec is None: return None
        _keys(spec, {"path", "sha256", "record"}, "invalid_worker_terminal_evidence")
        path = self.worker_output / "terminal.json"
        value, digest = _record(path)
        if (spec["path"] != str(path) or spec["sha256"] != digest or spec["record"] != value
                or load_terminal_or_crash(path) != value or value["native_action_limit"] != 1400):
            raise WorkerClientError("worker_terminal_binding_invalid")
        self.last_worker_terminal = deepcopy(spec)
        return value

    def _native_result(self, response, ticket):
        evaluator = response.get("evaluator")
        _keys(evaluator, {"native_result", "receipt", "terminal"}, "invalid_evaluator_response")
        raw = evaluator["native_result"]
        _keys(raw, {field.name for field in fields(NativeResult)}, "invalid_native_result_fields")
        result = NativeResult(**raw)
        reward, partial = result.native_reward, result.partial_score
        if (type(result.action_index) is not int or result.action_index != ticket
                or result.callback_verified is not True or result.checks_registered is not True
                or type(result.end_flag) is not bool or type(reward) not in (int,float)
                or not math.isfinite(reward) or not 0 <= reward <= 1
                or (partial is not None and (type(partial) not in (int,float) or not math.isfinite(partial) or not 0 <= partial <= 100))
                or result.evidence_id != f"{self._binding['claim_nonce']}:native-action:{ticket}"):
            raise WorkerClientError("unverified_native_result")
        spec = evaluator["receipt"]
        _keys(spec, {"path", "sha256"}, "invalid_native_receipt_binding")
        path = self.worker_output / f"action_{ticket:04d}.json"
        receipt, digest = _record(path)
        expected = {"schema": "robodojo-native-action-receipt-1", "scope": "evaluator_only", "model_input_eligible": False,
            "episode_evidence_id": self._binding["claim_nonce"], "authorization_verified_for_ticket": ticket,
            "native_action_index": ticket, "physics_dt_s": .004, "native_sim_step_calls": 10,
            "native_terminal_callback_calls": 1, "native_reward_callback_calls": 1, "native_result": raw}
        if (spec != {"path": str(path), "sha256": digest}
                or any(type(receipt.get(k)) is not type(v) or receipt.get(k) != v for k,v in expected.items())
                or receipt["native_physics_counter_after"] - receipt["native_physics_counter_before"] != 10
                or not math.isclose(receipt["physics_time_after_s"] - receipt["physics_time_before_s"], .04, abs_tol=1e-7, rel_tol=0)
                or receipt["native_callback_state"]["reward"] != reward
                or receipt["native_callback_state"]["end_flag"] != result.end_flag):
            raise WorkerClientError("native_receipt_accounting_invalid")
        terminal = self._terminal(evaluator["terminal"])
        ended = result.end_flag or ticket == 1400
        if (ended != (terminal is not None)
                or (terminal is not None and (terminal["confirmed_actions"] != ticket
                    or terminal["native_evidence"]["evidence_id"] != result.evidence_id
                    or terminal["native_evidence"]["native_reward"] != reward))):
            raise WorkerClientError("terminal_result_disagreement")
        timing = receipt.get("timing")
        names = ("policy_begin_monotonic_s","dispatch_started_monotonic_s",
            "native_callback_completed_monotonic_s","dispatch_completed_monotonic_s")
        _keys(timing,{"clock",*names},"invalid_native_wall_timing")
        values = [timing[key] for key in names]
        if (timing["clock"] != "time.monotonic" or any(type(v) not in (int,float) or not math.isfinite(v) or v < 0 for v in values)
                or values != sorted(values)):
            raise WorkerClientError("invalid_native_wall_timing")
        if ticket in self._verified_timings:
            if timing != self._verified_timings[ticket]: raise WorkerClientError("native_wall_timing_changed")
        else:
            if ticket != len(self._verified_timings)+1: raise WorkerClientError("native_wall_timing_sequence_gap")
            previous = self._verified_timings.get(ticket-1)
            if previous is not None and (timing[names[0]] != previous[names[0]]
                    or timing[names[1]] < previous[names[3]]):
                raise WorkerClientError("native_wall_timing_discontinuous")
            self._verified_timings[ticket] = deepcopy(timing)
        return result

    def dispatch(self, ticket: int, joint_action: dict) -> ActionReceipt:
        verified = None
        sent = False
        try:
            snapshot = self.lifecycle.snapshot()
            if (self.state != "RUNNING" or snapshot["state"] != "RUNNING"
                    or type(ticket) is not int or ticket != snapshot["in_flight"] or ticket != self._last_ticket+1):
                raise WorkerClientError("monotonic_reserved_parent_ticket_required")
            action = validate_joint_action(joint_action)
            self.check_binding()
            self._last_ticket = ticket  # No resend after an uncertain dispatch.
            sent = True
            response = self._request("action", {"ticket": ticket, "joint_action": action})
            verified = self._native_result(response, ticket)
            write_private_record(self.output / f"action_{ticket:04d}.json", {"scope": "evaluator_only", **response["evaluator"]})
            self._grippers = {arm: "open" if action[arm+"_ee_joint_state"][0] == 1 else "close" for arm in ARMS}
            measurement, _ = self._measurement(response, evaluator=True)
            if verified.end_flag or ticket == 1400: self.state = "TERMINAL"
            return ActionReceipt(measurement, verified)
        except BaseException as error:
            self._poison()
            # The worker commits its evaluator receipt before robot feedback.
            # A lost response/measurement error cannot erase that native truth.
            # This is evidence recovery only: never resend or query rewards.
            if verified is None and sent:
                try:
                    path = self.worker_output / f"action_{ticket:04d}.json"
                    receipt, digest = _record(path)
                    terminal = None
                    terminal_path = self.worker_output / "terminal.json"
                    if terminal_path.exists():
                        value, terminal_hash = _record(terminal_path)
                        terminal_spec = {"path": str(terminal_path), "sha256": terminal_hash, "record": value}
                        self._terminal(terminal_spec)
                        if value["status"] in ("success", "native_failure", "horizon_exhausted"):
                            terminal = terminal_spec
                    recovered = {"native_result": receipt["native_result"],
                        "receipt": {"path": str(path), "sha256": digest}, "terminal": terminal}
                    verified = self._native_result({"evaluator": recovered}, ticket)
                    if not (self.output / f"action_{ticket:04d}.json").exists():
                        write_private_record(self.output / f"action_{ticket:04d}.json", {
                            "scope": "evaluator_only", "recovered_after_rpc_failure": True, **recovered})
                except Exception:
                    pass  # Unverified/missing evidence remains an unconfirmed dispatch.
            if verified is not None:
                raise VerifiedActionFollowupError(verified) from None
            raise

    def abort(self) -> None:
        """Poison this channel; worker EOF handles its own immutable terminal truth."""
        self._poison()

    def close(self):
        if self.state in ("CLOSED", "FAILED"):
            self.channel.close(); return deepcopy(self.last_worker_terminal)
        try:
            response = self._request("close")
            _keys(response, {"state", "evaluator"}, "invalid_close_response")
            _keys(response["evaluator"], {"terminal"}, "invalid_close_evaluator")
            if response["state"] != "CLOSED": raise WorkerClientError("invalid_close_state")
            self._terminal(response["evaluator"]["terminal"])
            self.state = "CLOSED"
            return deepcopy(self.last_worker_terminal)
        except BaseException:
            self._poison(); raise
        finally:
            self.channel.close()
