#!/usr/bin/env python3
"""Pair an iPad with the existing Codex server over its private Unix socket.

Uses the protocol schema shipped with Codex CLI 0.153.4. This helper never
starts/stops a server, changes sandbox permissions, or reads auth.json.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import socket
import struct
import sys
import time


class CodexSocket:
    """Small RFC 6455 client for a local, trusted AF_UNIX transport only."""

    def __init__(self, path: Path, timeout: float = 20):
        self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.socket.settimeout(timeout)
        self.timeout = timeout
        self.buffer = bytearray()
        self.request_id = 0
        try:
            self.socket.connect(str(path))
            key = base64.b64encode(os.urandom(16)).decode()
            self.socket.sendall((
                "GET /rpc HTTP/1.1\r\nHost: localhost\r\n"
                "Upgrade: websocket\r\nConnection: Upgrade\r\n"
                f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n"
            ).encode())
            while b"\r\n\r\n" not in self.buffer:
                chunk = self.socket.recv(4096)
                if not chunk:
                    raise RuntimeError("Codex closed the WebSocket handshake")
                self.buffer.extend(chunk)
                if len(self.buffer) > 65536:
                    raise RuntimeError("Oversized WebSocket handshake")
            headers, remaining = bytes(self.buffer).split(b"\r\n\r\n", 1)
            self.buffer = bytearray(remaining)
            lines = headers.decode("ascii").split("\r\n")
            if lines[0].split()[1] != "101":
                raise RuntimeError(f"WebSocket handshake failed: {lines[0]}")
            fields = {name.lower(): value.strip() for name, value in
                      (line.split(":", 1) for line in lines[1:] if ":" in line)}
            expected = base64.b64encode(hashlib.sha1(
                (key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode()
            ).digest()).decode()
            if fields.get("sec-websocket-accept", "") != expected:
                raise RuntimeError("Invalid WebSocket handshake accept value")
            self.request("initialize", {
                "clientInfo": {"name": "gpt_field_ipad_setup", "version": "0.1.0"},
                "capabilities": {"experimentalApi": True},
            })
            self.send_json({"method": "initialized"})
        except BaseException:
            self.socket.close()
            raise

    def close(self):
        self.socket.close()

    def read_exact(self, size):
        while len(self.buffer) < size:
            chunk = self.socket.recv(max(4096, size - len(self.buffer)))
            if not chunk:
                raise RuntimeError("Codex closed the connection")
            self.buffer.extend(chunk)
        data = bytes(self.buffer[:size])
        del self.buffer[:size]
        return data

    def send_frame(self, opcode, payload):
        size = len(payload)
        header = bytes([0x80 | opcode])
        if size < 126:
            header += bytes([0x80 | size])
        elif size < 65536:
            header += bytes([0x80 | 126]) + struct.pack("!H", size)
        else:
            header += bytes([0x80 | 127]) + struct.pack("!Q", size)
        mask = os.urandom(4)
        masked = bytes(value ^ mask[index % 4] for index, value in enumerate(payload))
        self.socket.sendall(header + mask + masked)

    def send_json(self, value):
        self.send_frame(1, json.dumps(value).encode())

    def receive_json(self):
        message = bytearray()
        while True:
            first, second = self.read_exact(2)
            opcode, size = first & 15, second & 127
            if size == 126:
                size = struct.unpack("!H", self.read_exact(2))[0]
            elif size == 127:
                size = struct.unpack("!Q", self.read_exact(8))[0]
            if size > 8 * 1024 * 1024 or len(message) + size > 8 * 1024 * 1024:
                raise RuntimeError("Oversized WebSocket message")
            mask = self.read_exact(4) if second & 128 else None
            payload = self.read_exact(size)
            if mask:
                payload = bytes(value ^ mask[index % 4] for index, value in enumerate(payload))
            if opcode == 8:
                raise RuntimeError("Codex closed the WebSocket")
            if opcode == 9:
                self.send_frame(10, payload)
                continue
            if opcode == 10:
                continue
            if opcode not in (0, 1):
                raise RuntimeError("Unexpected non-text WebSocket message")
            message.extend(payload)
            if first & 128:
                return json.loads(message)

    def request(self, method, params=None):
        self.request_id += 1
        request_id = self.request_id
        message = {"id": request_id, "method": method}
        if params is not None:
            message["params"] = params
        self.send_json(message)
        deadline = time.monotonic() + self.timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"Timed out waiting for {method}")
            self.socket.settimeout(remaining)
            response = self.receive_json()
            if response.get("id") != request_id:
                continue  # Never print unrelated conversation notifications.
            if "error" in response:
                error = response["error"]
                raise RuntimeError(f"{method}: {error.get('message', 'request failed')}")
            return response.get("result", {})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("status", "enable", "pair", "disable"))
    codex_home = Path(os.environ.get("CODEX_HOME", str(Path.home() / ".codex")))
    parser.add_argument("--socket", type=Path,
                        default=codex_home / "app-server-control/app-server-control.sock")
    parser.add_argument("--timeout", type=float, default=20)
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    client = CodexSocket(args.socket, args.timeout)
    try:
        if args.command == "enable":
            client.request("remoteControl/enable", {"ephemeral": False})
            deadline = time.monotonic() + args.timeout
            while True:
                result = client.request("remoteControl/status/read")
                if result.get("status") != "connecting" or time.monotonic() >= deadline:
                    break
                time.sleep(0.5)
        elif args.command == "disable":
            result = client.request("remoteControl/disable", {"ephemeral": False})
        elif args.command == "pair":
            status = client.request("remoteControl/status/read")
            if status.get("status") != "connected":
                raise RuntimeError("Remote control is not connected; run status or enable first")
            result = client.request("remoteControl/pairing/start", {"manualCode": True})
            # Print only the code the user enters and its expiry, not the raw pairing token.
            result = {key: result.get(key) for key in ("manualPairingCode", "expiresAt")}
            if not result["manualPairingCode"]:
                raise RuntimeError("The server did not return a manual pairing code")
        else:
            result = client.request("remoteControl/status/read")
        if args.command != "pair":
            result = {key: result.get(key) for key in ("status", "serverName")}
        print(json.dumps(result, indent=2))
        if args.command == "enable" and result.get("status") != "connected":
            return 1
        return 0
    finally:
        client.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
