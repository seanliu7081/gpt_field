#!/usr/bin/env python3
"""Bound a container child to its NVML PID using GPU4 context ON/OFF/ON.

Importing this module does not load CUDA. The child must explicitly construct
PrimaryContext after authorization; no Isaac Sim import occurs during calibration.
"""
from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path
import socket
import time
import uuid

GPU_UUID = "GPU-e1a460e9-e272-b227-222e-0dbb77c248c5"


class IdentityError(RuntimeError):
    pass


class Channel:
    def __init__(self, connection: socket.socket):
        self.socket = connection
        self.buffer = bytearray()

    def send(self, message: dict, deadline: float):
        if time.monotonic() >= deadline:
            raise IdentityError("Identity command deadline expired")
        raw = json.dumps(message, allow_nan=False, separators=(",", ":")).encode() + b"\n"
        if len(raw) > 4096:
            raise IdentityError("Identity frame is too large")
        self.socket.settimeout(max(0.001, deadline - time.monotonic()))
        self.socket.sendall(raw)

    def receive(self, deadline: float) -> dict:
        while b"\n" not in self.buffer:
            if time.monotonic() >= deadline or len(self.buffer) >= 4096:
                raise IdentityError("Identity handshake timed out or exceeded frame limit")
            self.socket.settimeout(deadline - time.monotonic())
            data = self.socket.recv(4096 - len(self.buffer))
            if not data:
                raise IdentityError("Identity channel closed")
            self.buffer.extend(data)
        raw, _, rest = self.buffer.partition(b"\n")
        self.buffer = bytearray(rest)
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise IdentityError("Identity message must be an object")
        return value


class PrimaryContext:
    def __init__(self):
        self.driver = ctypes.CDLL("libcuda.so.1")
        prototypes = {
            "cuInit": [ctypes.c_uint],
            "cuDeviceGet": [ctypes.POINTER(ctypes.c_int), ctypes.c_int],
            "cuDeviceGetUuid": [ctypes.c_void_p, ctypes.c_int],
            "cuDeviceGetPCIBusId": [ctypes.c_void_p, ctypes.c_int, ctypes.c_int],
            "cuDevicePrimaryCtxRetain": [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int],
            "cuCtxSetCurrent": [ctypes.c_void_p],
            "cuDevicePrimaryCtxRelease_v2": [ctypes.c_int],
        }
        for name, arguments in prototypes.items():
            function = getattr(self.driver, name)
            function.argtypes = arguments
            function.restype = ctypes.c_int
        self.device = ctypes.c_int()
        self.context = ctypes.c_void_p()
        self.retained = False
        self.check("cuInit", 0)
        self.check("cuDeviceGet", ctypes.byref(self.device), 4)
        raw_uuid = (ctypes.c_ubyte * 16)()
        self.check("cuDeviceGetUuid", ctypes.byref(raw_uuid), self.device)
        self.gpu_uuid = "GPU-" + str(uuid.UUID(bytes=bytes(raw_uuid)))
        pci = ctypes.create_string_buffer(64)
        self.check("cuDeviceGetPCIBusId", pci, len(pci), self.device)
        self.pci_bus_id = pci.value.decode()
        if self.gpu_uuid != GPU_UUID or not self.pci_bus_id.lower().endswith(":81:00.0"):
            raise IdentityError("CUDA ordinal4 does not identify the authorized physical GPU")

    def check(self, name, *args):
        result = getattr(self.driver, name)(*args)
        if result != 0:
            raise IdentityError(f"CUDA identity operation {name} failed with code {result}")

    def retain(self):
        if self.retained:
            raise IdentityError("Primary context is already retained")
        self.check("cuDevicePrimaryCtxRetain", ctypes.byref(self.context), self.device)
        self.retained = True
        self.check("cuCtxSetCurrent", self.context)

    def release(self):
        if self.retained:
            self.check("cuCtxSetCurrent", None)
            self.check("cuDevicePrimaryCtxRelease_v2", self.device)
            self.retained = False
            self.context = ctypes.c_void_p()


def own_start_ticks() -> int:
    return int(Path("/proc/self/stat").read_text().rsplit(") ", 1)[1].split()[19])


def child_calibration(output: Path, progress) -> PrimaryContext:
    """Return a retained context only after the parent verifies ON/OFF/ON."""
    connection = socket.socket(fileno=int(os.environ["UC71_A1_IDENTITY_FD"]))
    channel = Channel(connection)
    nonce = os.environ["UC71_A1_IDENTITY_NONCE"]
    deadline = time.monotonic() + 30
    context = PrimaryContext()
    identity = {"nonce": nonce, "child_pid": os.getpid(), "child_start_ticks": own_start_ticks(),
                "gpu_uuid": context.gpu_uuid, "cuda_ordinal": 4, "pci_bus_id": context.pci_bus_id}
    try:
        for phase, command in [("on1", "release"), ("off", "recreate"), ("on2", "continue")]:
            if phase in {"on1", "on2"}:
                context.retain()
            else:
                context.release()
            progress("cuda_identity_" + phase, **identity, context_retained=context.retained)
            channel.send({**identity, "phase": phase, "context_retained": context.retained}, deadline)
            reply = channel.receive(min(deadline, time.monotonic() + 10))
            if reply.get("nonce") != nonce or reply.get("command") != command:
                raise IdentityError("Parent identity challenge command differs from the expected phase")
            if command == "continue":
                if type(reply.get("nvml_pid")) is not int or reply["nvml_pid"] < 1:
                    raise IdentityError("Parent did not bind an NVML PID")
                record = {**identity, "nvml_pid": reply["nvml_pid"], "verified_on_off_on": True,
                          "retained_primary_context_through_startup": True}
                (output / "cuda_identity_child.json").write_text(json.dumps(record, indent=2) + "\n")
                progress("cuda_identity_verified", nvml_pid=reply["nvml_pid"])
        return context
    except BaseException:
        context.release()
        raise
    finally:
        connection.close()


def target_candidate(snapshot: list[dict], *, previous_pid: int | None, off: bool) -> int | None:
    """Pure validation: another GPU/process or changed host PID is ambiguous."""
    target = [g for g in snapshot if g["uuid"] == GPU_UUID]
    if len(target) != 1 or target[0]["index"] != 4:
        raise IdentityError("Physical GPU identity changed during calibration")
    if any(g["processes"] for g in snapshot if g["uuid"] != GPU_UUID):
        raise IdentityError("Another GPU became active during single-GPU identity calibration")
    entries = target[0]["processes"]
    if not entries:
        return None
    if len(entries) != 1 or "C" not in str(entries[0].get("type", "")):
        raise IdentityError("Target GPU identity is ambiguous or is not a compute context")
    pid = entries[0].get("pid")
    if type(pid) is not int or pid < 1 or (previous_pid is not None and pid != previous_pid):
        raise IdentityError("The NVML PID changed during the same-child identity challenge")
    return pid


def parent_calibration(connection: socket.socket, *, proc, initial: dict, nonce: str,
                       output: Path, gpu_snapshot, process_record) -> dict:
    """Verify presence, disappearance and same-PID return within 30 seconds."""
    channel = Channel(connection)
    deadline = time.monotonic() + 30
    nvml_pid = None
    events = []
    with (output / "cuda_identity_parent.jsonl").open("x") as log:
        for phase, command in [("on1", "release"), ("off", "recreate"), ("on2", "continue")]:
            phase_deadline = min(deadline, time.monotonic() + 10)
            message = channel.receive(phase_deadline)
            live = process_record(proc.pid)
            if (live is None or live["start_ticks"] != initial["start_ticks"]
                    or message.get("child_pid") != proc.pid
                    or message.get("child_start_ticks") != initial["start_ticks"]
                    or message.get("nonce") != nonce or message.get("gpu_uuid") != GPU_UUID
                    or message.get("cuda_ordinal") != 4 or message.get("phase") != phase
                    or message.get("context_retained") is not (phase != "off")):
                raise IdentityError("Child identity or primary-context phase does not match")
            stable = 0
            last = None
            while True:
                remaining = phase_deadline - time.monotonic()
                if remaining <= 0 or proc.poll() is not None:
                    raise IdentityError("Primary-context identity challenge timed out or child exited")
                snapshot = gpu_snapshot(timeout_s=min(3, remaining))
                candidate = target_candidate(snapshot, previous_pid=nvml_pid, off=phase == "off")
                if phase == "on1" and candidate is not None and nvml_pid is None:
                    nvml_pid = candidate
                event = {"phase": phase, "monotonic_s": time.monotonic(), "candidate": candidate,
                         "child": message, "gpus": snapshot}
                log.write(json.dumps(event) + "\n"); log.flush()
                events.append(event)
                desired = candidate is None if phase == "off" else candidate is not None
                stable = stable + 1 if desired and candidate == last else 1 if desired else 0
                last = candidate
                if stable >= 2:
                    if phase == "on1":
                        nvml_pid = candidate
                    break
                time.sleep(min(0.1, max(0, phase_deadline - time.monotonic())))
            channel.send({"nonce": nonce, "command": command, "nvml_pid": nvml_pid}, phase_deadline)
        result = {"nonce": nonce, "child_pid": proc.pid, "child_start_ticks": initial["start_ticks"],
                  "nvml_pid": nvml_pid, "gpu_uuid": GPU_UUID, "verified_on_off_on": True,
                  "verified_samples": len(events), "method": "same-child GPU4 primary-context presence/disappearance/same-PID return"}
        (output / "cuda_identity_binding.json").write_text(json.dumps(result, indent=2) + "\n")
        return result
