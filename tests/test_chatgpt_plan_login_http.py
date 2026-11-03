"""Loopback HTTP regressions; no real accounts or OAuth requests are used."""

from http.client import HTTPConnection
import importlib.util
from pathlib import Path
import socket
import threading
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import pytest


@pytest.fixture
def login_module():
    path = Path(__file__).resolve().parents[1] / "scripts/chatgpt_plan_login.py"
    spec = importlib.util.spec_from_file_location("chatgpt_plan_login_http_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def listener(tmp_path, monkeypatch, request, login_module):
    module = login_module
    browser_port = getattr(request, "param", None)

    ready = threading.Event()
    handling_idle_connection = threading.Event()
    bound = {}
    outcomes = []

    class ObservedServer(module.LoginHTTPServer):
        def __init__(self, address, handler):
            class ObservedHandler(handler):
                def handle(self):
                    handling_idle_connection.set()
                    super().handle()

            # Preserve the helper's exact Host check while avoiding real port 1455.
            assert address == ("127.0.0.1", 1455)
            super().__init__((address[0], 0), ObservedHandler)
            bound["port"] = self.server_address[1]
            ready.set()

    monkeypatch.setattr(module, "LoginHTTPServer", ObservedServer)
    monkeypatch.setattr(module, "OpenAIAuthService", lambda: object())
    store = module.TokenStore(tmp_path / "credentials")

    def run():
        try:
            outcomes.append(module.login(store, profile=None, new_account=True,
                                         port=1455, browser_port=browser_port, timeout=2))
        except Exception as exc:
            outcomes.append(exc)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    assert ready.wait(3), "The login listener did not start"
    idle = socket.create_connection(("127.0.0.1", bound["port"]), timeout=1)
    assert handling_idle_connection.wait(3), "Idle connection was not accepted"

    def request_page(host="127.0.0.1:1455", path="/login", *, include_headers=False):
        connection = HTTPConnection("127.0.0.1", bound["port"], timeout=1)
        try:
            connection.request("GET", path, headers={"Host": host})
            response = connection.getresponse()
            body = response.read().decode("utf-8")
            if include_headers:
                return response.status, dict(response.getheaders()), body
            return response.status, body
        finally:
            connection.close()

    try:
        yield SimpleNamespace(request=request_page, thread=thread, outcomes=outcomes,
                              module=module, store=store)
    finally:
        idle.close()
        thread.join(4)
        assert not thread.is_alive(), "The login listener did not stop"


def test_idle_preconnection_does_not_block_login_page(listener):
    status, body = listener.request()
    assert status == 200
    assert "GPT Field Robot Planning" in body
    assert '<a href="/authorize">Continue with ChatGPT</a>' in body
    assert listener.store.status()["connected"] is False


def test_threaded_listener_preserves_exact_host_check(listener):
    status, body = listener.request(host="localhost:1455")
    assert status == 400
    assert "Use the exact loopback sign-in URL" in body
    assert "Continue with ChatGPT" not in body


def test_idle_preconnection_does_not_disable_login_deadline(listener):
    # The first socket remains open throughout this wait; no client data arrives.
    listener.thread.join(4)
    assert not listener.thread.is_alive()
    assert len(listener.outcomes) == 1
    assert isinstance(listener.outcomes[0], listener.module.AuthError)
    assert "Browser authorization timed out" in str(listener.outcomes[0])


@pytest.mark.parametrize("listener", [1456], indirect=True)
def test_forwarded_browser_port_controls_host_and_oauth_callback(listener):
    status, body = listener.request(host="127.0.0.1:1456")
    assert status == 200
    assert "Continue with ChatGPT" in body

    for wrong_host in ("127.0.0.1:1455", "localhost:1456"):
        status, body = listener.request(host=wrong_host)
        assert status == 400
        assert "Continue with ChatGPT" not in body

    status, headers, _ = listener.request(host="127.0.0.1:1456", path="/authorize",
                                         include_headers=True)
    assert status == 302
    authorization = urlsplit(headers["Location"])
    assert authorization.scheme == "https"
    assert authorization.netloc == "auth.openai.com"
    query = parse_qs(authorization.query)
    assert query["redirect_uri"] == ["http://127.0.0.1:1456/auth/callback"]
    assert listener.store.status()["connected"] is False


@pytest.mark.parametrize("browser_port", [-1, 0, 65536])
def test_invalid_browser_port_is_rejected(login_module, tmp_path, browser_port):
    store = login_module.TokenStore(tmp_path / "credentials")
    with pytest.raises(login_module.AuthError):
        login_module.login(store, profile=None, new_account=True, port=1455,
                           browser_port=browser_port, timeout=2)


def test_cli_passes_browser_port_separately_from_listener(login_module, tmp_path,
                                                        monkeypatch):
    calls = []

    def fake_login(store, **kwargs):
        calls.append(kwargs)
        return {"connected": False}

    monkeypatch.setattr(login_module, "login", fake_login)
    assert login_module.main(["--store-dir", str(tmp_path / "credentials"),
                              "login", "--new-account", "--port", "1455",
                              "--browser-port", "1456", "--timeout", "120"]) == 0
    assert calls == [{"profile": None, "new_account": True, "port": 1455,
                      "browser_port": 1456, "timeout": 120}]
