from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import pytest

from robodojo_uc71.ipc import JsonLineChannel, ProtocolError, RemoteError, worker_environment


def pair(**kwargs):
    a, b = socket.socketpair()
    return JsonLineChannel(a, role="parent", **kwargs), JsonLineChannel(b, role="worker", **kwargs)


def test_roundtrip_strict_sequence_and_sanitized_remote_error():
    parent, worker = pair()
    def serve():
        first = worker.receive_request()
        assert first["seq"] == 1 and first["method"] == "ready"
        worker.respond(first["seq"], {"state": "READY"})
        second = worker.receive_request()
        assert second["seq"] == 2
        worker.respond(second["seq"], error_code="native_worker_error")
        worker.close()
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(serve)
        assert parent.call("ready") == {"state": "READY"}
        with pytest.raises(RemoteError, match="native_worker_error"):
            parent.call("step", {"action": 1})
        assert parent.state == "IDLE" and parent.next_sequence == 3
        future.result()
    parent.close()


@pytest.mark.parametrize("raw", [
    b'{"version":1,"kind":"response","seq":2,"ok":true,"payload":{},"error":null}\n',
    b'{"version":1,"version":1,"kind":"response","seq":1,"ok":true,"payload":{},"error":null}\n',
    b'{"version":1,"kind":"response","seq":1,"ok":true,"payload":{"x":NaN},"error":null}\n',
    b'{"version":1,"kind":"response","seq":true,"ok":true,"payload":{},"error":null}\n',
    b'{"version":1,"kind":"response","seq":1,"ok":true,"payload":{"x":1e999},"error":null}\n',
    b'not-json\n',
])
def test_malformed_or_out_of_order_response_closes_connection(raw):
    a, b = socket.socketpair()
    parent = JsonLineChannel(a, role="parent")
    b.sendall(raw)
    with pytest.raises(ProtocolError):
        parent.call("ready")
    assert parent.state == "CLOSED"
    b.close()


def test_frame_limit_and_deadline_are_bounded():
    a, b = socket.socketpair()
    worker = JsonLineChannel(a, role="worker", max_bytes=128, timeout_s=0.03)
    b.sendall(b"x"*128)
    with pytest.raises(ProtocolError, match="byte limit"):
        worker.receive_request()
    b.close()
    parent, worker = pair(timeout_s=0.03)
    with pytest.raises(ProtocolError, match="timed out"):
        parent.call("ready")
    assert parent.state == "CLOSED"
    worker.close()


def test_worker_state_rejects_second_receive_before_response():
    a, b = socket.socketpair()
    worker = JsonLineChannel(a, role="worker")
    b.sendall(json.dumps({"version":1,"kind":"request","seq":1,"method":"ready","payload":{}}).encode()+b"\n")
    request = worker.receive_request()
    with pytest.raises(ProtocolError):
        worker.receive_request()
    with pytest.raises(ProtocolError):
        worker.respond(2, {})
    worker.respond(request["seq"], {})
    worker.close(); b.close()


def test_worker_environment_drops_credentials_and_rejects_secret_override():
    env = worker_environment({"PATH":"/bin", "OPENAI_API_KEY":"secret", "HF_TOKEN":"secret",
                              "CHATGPT_TOKEN":"secret", "AWS_SECRET_ACCESS_KEY":"secret"})
    assert env == {"PATH":"/bin"}
    with pytest.raises(ProtocolError):
        worker_environment({}, overrides={"OPENAI_API_KEY":"secret"})


def test_inherited_socket_protocol_survives_worker_stdout_dup2():
    a, b = socket.socketpair()
    parent = JsonLineChannel(a, role="parent", timeout_s=5)
    source_root = str(Path(__file__).resolve().parents[1] / "src")
    code = '''
import os,sys
from robodojo_uc71.ipc import JsonLineChannel
channel=JsonLineChannel.from_fd(int(sys.argv[1]),role="worker",timeout_s=5)
os.close(int(sys.argv[1]))
fd=os.open(os.devnull,os.O_WRONLY)
os.dup2(fd,1);os.dup2(fd,2);os.close(fd)
request=channel.receive_request()
channel.respond(request["seq"],{"state":"READY","pid":os.getpid()})
channel.close()
'''
    proc = subprocess.Popen([sys.executable,"-c",code,str(b.fileno())],pass_fds=(b.fileno(),),
                            env=worker_environment(os.environ,overrides={"PYTHONPATH":source_root}))
    b.close()
    try:
        answer = parent.call("ready")
        assert answer == {"state":"READY", "pid":proc.pid}
        assert proc.wait(timeout=5) == 0
    finally:
        parent.close()
        if proc.poll() is None:
            proc.kill(); proc.wait()


def test_idle_worker_survives_wait_longer_than_frame_timeout():
    parent, worker = pair(timeout_s=.03)
    def serve():
        request = worker.receive_request()
        worker.respond(request["seq"], {"idle_survived":True})
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(serve)
        time.sleep(.09)
        assert parent.call("ready", timeout_s=1) == {"idle_survived":True}
        future.result(timeout=1)
    parent.close(); worker.close()


def test_partial_frame_and_explicit_idle_deadlines_remain_bounded():
    a,b=socket.socketpair()
    worker=JsonLineChannel(a,role="worker",timeout_s=.03)
    b.sendall(b"{")
    with pytest.raises(ProtocolError,match="timed out"):
        worker.receive_request()
    b.close()
    a,b=socket.socketpair()
    worker=JsonLineChannel(a,role="worker",timeout_s=1)
    with pytest.raises(ProtocolError,match="timed out"):
        worker.receive_request(timeout_s=.03)
    b.close()


def test_idle_peer_eof_closes_worker():
    a,b=socket.socketpair()
    worker=JsonLineChannel(a,role="worker")
    b.close()
    with pytest.raises(ProtocolError,match="peer closed"):
        worker.receive_request()
    assert worker.state=="CLOSED"
