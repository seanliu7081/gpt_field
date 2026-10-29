"""Offline protocol tests; never connect to the actual Codex server."""

import base64
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "ipad_codex.py"
GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


class FakePeer:
    def __init__(self, connection):
        self.connection = connection
        self.buffer = bytearray()
        self.requests = []

    def read_exact(self, size):
        while len(self.buffer) < size:
            data = self.connection.recv(4096)
            if not data:
                raise AssertionError("Client closed before the expected frame")
            self.buffer.extend(data)
        data = bytes(self.buffer[:size])
        del self.buffer[:size]
        return data

    def handshake(self, *, valid=True):
        while b"\r\n\r\n" not in self.buffer:
            data = self.connection.recv(4096)
            if not data:
                raise AssertionError("Client closed during handshake")
            self.buffer.extend(data)
        header, rest = bytes(self.buffer).split(b"\r\n\r\n", 1)
        self.buffer = bytearray(rest)
        lines = header.decode("ascii").split("\r\n")
        assert lines[0] == "GET /rpc HTTP/1.1", lines[0]
        fields = dict(line.split(": ", 1) for line in lines[1:])
        assert fields["Host"] == "localhost"
        assert fields["Upgrade"].lower() == "websocket"
        assert fields["Sec-WebSocket-Version"] == "13"
        key = fields["Sec-WebSocket-Key"]
        assert len(base64.b64decode(key)) == 16
        accept = base64.b64encode(hashlib.sha1((key + GUID).encode()).digest()).decode()
        if not valid:
            accept = "deliberately-invalid"
        self.connection.sendall(("HTTP/1.1 101 Switching Protocols\r\n"
            "Upgrade: websocket\r\nConnection: Upgrade\r\n"
            f"Sec-WebSocket-Accept: {accept}\r\n\r\n").encode())

    def receive(self):
        first, second = self.read_exact(2)
        assert first & 128, "Client frames must be final"
        assert second & 128, "Client frames must be masked"
        length = second & 127
        if length == 126:
            length = struct.unpack("!H", self.read_exact(2))[0]
        elif length == 127:
            length = struct.unpack("!Q", self.read_exact(8))[0]
        mask = self.read_exact(4)
        payload = bytes(value ^ mask[index % 4]
                        for index, value in enumerate(self.read_exact(length)))
        return first & 15, payload

    def request(self, method):
        opcode, payload = self.receive()
        assert opcode == 1, opcode
        request = json.loads(payload)
        assert request["method"] == method, request
        self.requests.append(request)
        return request

    def frame(self, payload, *, opcode=1, final=True):
        header = bytes([(128 if final else 0) | opcode])
        size = len(payload)
        if size < 126:
            header += bytes([size])
        elif size < 65536:
            header += bytes([126]) + struct.pack("!H", size)
        else:
            header += bytes([127]) + struct.pack("!Q", size)
        self.connection.sendall(header + payload)

    def send(self, value):
        self.frame(json.dumps(value).encode())

    def result(self, request, value):
        self.send({"id": request["id"], "result": value})

    def initialize(self):
        request = self.request("initialize")
        assert request["params"]["capabilities"]["experimentalApi"] is True
        assert request["params"]["clientInfo"]["name"] == "gpt_field_ipad_setup"
        self.result(request, {"userAgent": "offline-fixture"})
        initialized = self.request("initialized")
        assert "id" not in initialized


class IpadCodexTests(unittest.TestCase):
    @contextmanager
    def server(self, handler, *, valid_handshake=True):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.sock"
            listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            listener.settimeout(5)
            listener.bind(str(path))
            listener.listen(1)
            errors = []

            def serve():
                try:
                    connection, _ = listener.accept()
                    with connection:
                        connection.settimeout(3)
                        peer = FakePeer(connection)
                        peer.handshake(valid=valid_handshake)
                        if valid_handshake:
                            peer.initialize()
                            handler(peer)
                except BaseException as exc:
                    errors.append(exc)

            thread = threading.Thread(target=serve, daemon=True)
            thread.start()
            try:
                yield path
            finally:
                thread.join(5)
                listener.close()
                self.assertFalse(thread.is_alive(), "Fixture server did not exit")
                if errors:
                    raise errors[0]

    def invoke(self, command, path):
        return subprocess.run([sys.executable, str(SCRIPT), command, "--socket", str(path),
                               "--timeout", "2"], capture_output=True, text=True, timeout=6)

    def test_status_handshake_notifications_fragmentation_and_ping(self):
        def handler(peer):
            request = peer.request("remoteControl/status/read")
            peer.send({"method": "conversation/private", "params": {"text": "DO_NOT_PRINT_PRIVATE"}})
            payload = json.dumps({"id": request["id"], "result": {
                "status": "connected", "serverName": "Fixture", "extra": "x" * 200}}).encode()
            peer.frame(payload[:160], final=False)
            peer.frame(b"heartbeat", opcode=9)
            self.assertEqual(peer.receive(), (10, b"heartbeat"))
            peer.frame(payload[160:], opcode=0)

        with self.server(handler) as path:
            result = self.invoke("status", path)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), {"status": "connected", "serverName": "Fixture"})
        self.assertNotIn("DO_NOT_PRINT_PRIVATE", result.stdout + result.stderr)

    def test_enable_requests_persistent_mode_then_reads_connected_status(self):
        def handler(peer):
            request = peer.request("remoteControl/enable")
            self.assertEqual(request["params"], {"ephemeral": False})
            peer.result(request, {})
            request = peer.request("remoteControl/status/read")
            peer.result(request, {"status": "connecting", "serverName": "Fixture"})
            request = peer.request("remoteControl/status/read")
            peer.result(request, {"status": "connected", "serverName": "Fixture"})

        with self.server(handler) as path:
            result = self.invoke("enable", path)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["status"], "connected")

    def test_pair_outputs_manual_code_and_expiry_but_never_raw_token(self):
        def handler(peer):
            request = peer.request("remoteControl/status/read")
            peer.result(request, {"status": "connected"})
            request = peer.request("remoteControl/pairing/start")
            self.assertEqual(request["params"], {"manualCode": True})
            peer.result(request, {"manualPairingCode": "123-456", "expiresAt": 1700000000,
                "pairingToken": "RAW_TOKEN_MUST_STAY_PRIVATE", "pairingUrl": "PRIVATE_LINK"})

        with self.server(handler) as path:
            result = self.invoke("pair", path)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), {"manualPairingCode": "123-456", "expiresAt": 1700000000})
        self.assertNotIn("RAW_TOKEN_MUST_STAY_PRIVATE", result.stdout + result.stderr)
        self.assertNotIn("PRIVATE_LINK", result.stdout + result.stderr)

    def test_rpc_error_and_closed_connections_fail_cleanly(self):
        for kind in ("rpc_error", "close_frame", "eof"):
            with self.subTest(kind=kind):
                def handler(peer):
                    request = peer.request("remoteControl/status/read")
                    if kind == "rpc_error":
                        peer.send({"id": request["id"], "error": {"code": -32000, "message": "fixture refused"}})
                    elif kind == "close_frame":
                        peer.frame(struct.pack("!H", 1000), opcode=8)
                    # Returning closes the Unix stream for the EOF case.

                with self.server(handler) as path:
                    result = self.invoke("status", path)
                self.assertEqual(result.returncode, 1)
                self.assertEqual(result.stdout, "")
                self.assertIn("Error:", result.stderr)
                self.assertNotIn("Traceback", result.stderr)

    def test_invalid_websocket_accept_is_rejected_before_initialization(self):
        with self.server(lambda peer: None, valid_handshake=False) as path:
            result = self.invoke("status", path)
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout, "")
        self.assertIn("Invalid WebSocket handshake accept", result.stderr)


if __name__ == "__main__":
    unittest.main()
