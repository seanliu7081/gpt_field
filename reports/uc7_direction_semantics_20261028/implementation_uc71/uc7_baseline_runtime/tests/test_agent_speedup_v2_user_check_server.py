"""Offline HTTP contract tests: real review artifacts, no inference or simulator."""

import http.client
import json
import threading
from urllib.parse import urlsplit

import pytest

from gpt_field.agent_speedup_v2_user_check import ReviewStore
from gpt_field.agent_speedup_v2_user_check_server import MAX_BODY_BYTES, ReviewServer, serve


def _plan():
    return {
        "scene": "A small object near a support surface.",
        "parts": [
            {"id": "A", "text": "object: whole body", "move": "free", "w": 1, "x": 2},
            {"id": "B", "text": "support: upper surface", "move": "fixed", "w": 3, "x": 4},
        ],
        "hold_true": [],
        "steps": [
            {"do": "place_on", "obj": "A", "grip": [], "ref": "B", "sense": "any",
             "keep": [{"c": "upright", "p": "A"}], "until": []}
        ],
    }


@pytest.fixture
def review_store(tmp_path):
    return ReviewStore.create(
        tmp_path / "review", task_key="test_task", episode_index=0,
        task_prompt="Place the object on its support.",
        initial_observation={"rgb": "offline-fixture", "state": [0, 0, 0]},
        plan=_plan(), runtime_sha256={"source.py": "0" * 64},
        preparation={"review_images": []},
    )


@pytest.fixture
def server(review_store):
    with ReviewServer(review_store, port=0) as instance:
        yield instance


def request(server, path, *, method="GET", data=None, body=None, headers=None, auth=True):
    actual_headers = {}
    if auth:
        actual_headers["Authorization"] = f"Bearer {server.token}"
    if data is not None:
        body = json.dumps(data).encode()
        actual_headers["Content-Type"] = "application/json"
    actual_headers.update(headers or {})
    connection = http.client.HTTPConnection("127.0.0.1", server.port, timeout=3)
    try:
        connection.request(method, path, body=body, headers=actual_headers)
        response = connection.getresponse()
        raw = response.read()
        try:
            result = json.loads(raw)
        except ValueError:
            result = raw.decode()
        return response.status, dict(response.getheaders()), result
    finally:
        connection.close()


def test_session_url_hides_token_in_fragment(server):
    parsed = urlsplit(server.url)
    assert parsed.hostname == "127.0.0.1"
    assert not parsed.query
    assert parsed.fragment == "token=" + server.token
    assert len(server.token) >= 40
    status, headers, page = request(server, "/", auth=False)
    assert status == 200
    assert server.token not in page
    assert "__CSP_NONCE__" not in page
    assert "frame-ancestors 'none'" in headers["Content-Security-Policy"]
    assert headers["Cache-Control"] == "no-store"
    assert headers["Referrer-Policy"] == "no-referrer"
    assert 'id="step-editor"' in page
    assert "textarea" not in page


def test_api_reads_are_authenticated(server):
    for path in ("/api/review", "/api/schema"):
        assert request(server, path, auth=False)[0] == 401
        assert request(server, path, headers={"Authorization": "Bearer wrong"})[0] == 401
        assert request(server, path)[0] == 200


@pytest.mark.parametrize("host", ["evil.example", "127.0.0.1.evil.example", "localhost", "127.0.0.1:1"])
def test_rebinding_hosts_are_rejected(server, host):
    assert request(server, "/api/review", headers={"Host": host})[0] == 403
    assert request(server, "/", headers={"Host": host})[0] == 403


@pytest.mark.parametrize("origin", ["https://evil.example", "null", "http://127.0.0.1:1"])
def test_cross_origin_requests_are_rejected(server, origin):
    assert request(server, "/api/review", headers={"Origin": origin})[0] == 403
    assert request(
        server, "/api/approve", method="POST",
        data={"expected_revision": 0, "reviewed_steps": [0]}, headers={"Origin": origin},
    )[0] == 403


def test_same_origin_allowed_and_cross_site_fetch_rejected(server):
    headers = {"Origin": f"http://127.0.0.1:{server.port}", "Sec-Fetch-Site": "same-origin"}
    assert request(server, "/api/review", headers=headers)[0] == 200
    assert request(server, "/api/review", headers={"Sec-Fetch-Site": "cross-site"})[0] == 403


@pytest.mark.parametrize("path", ["/unknown", "/api/review?file=secret", "/../../etc/passwd", "/api/run"])
def test_no_arbitrary_routes_or_execution_endpoint(server, path):
    assert request(server, path)[0] == 404
    assert request(server, path, method="POST", data={})[0] == 404


def test_schema_has_real_v2_enums_and_capabilities(server):
    status, _, schema = request(server, "/api/schema")
    assert status == 200
    assert schema["implemented_do"] == ["hold", "turn", "place_on", "release"]
    step = schema["plan_schema"]["properties"]["steps"]["items"]["properties"]
    assert step["keep"]["items"]["properties"]["c"]["enum"] == ["upright", "clear_of"]
    assert step["grip"]["maxItems"] == 2
    assert schema["allowed_until"]["hold"] == []


@pytest.mark.parametrize("body", [b"{", b"[]", b"\xff", b'{"x":NaN}', b'{"x":1,"x":2}'])
def test_malformed_json_cannot_change_review(server, review_store, body):
    status, _, _ = request(
        server, "/api/update", method="POST", body=body, headers={"Content-Type": "application/json"}
    )
    assert status == 400
    assert review_store.load()["revision"] == 0


def test_oversize_and_wrong_content_type_rejected(server):
    assert request(server, "/api/update", method="POST", body=b"{}", headers={"Content-Type": "text/plain"})[0] == 415
    assert request(server, "/api/update", method="POST", body=b"{}",
                   headers={"Content-Type": "application/json", "Content-Length": str(MAX_BODY_BYTES + 1)})[0] == 413


@pytest.mark.parametrize(
    "payload",
    [
        {"expected_revision": True, "reviewed_steps": [0]},
        {"expected_revision": 0, "reviewed_steps": [True]},
        {"expected_revision": 0, "reviewed_steps": [0, 0]},
        {"expected_revision": 0, "reviewed_steps": [-1]},
        {"expected_revision": 0, "reviewed_steps": [0], "run": True},
    ],
)
def test_approval_request_requires_exact_fields_and_types(server, payload):
    assert request(server, "/api/approve", method="POST", data=payload)[0] == 400


def test_semantic_edits_are_saved_invalid_but_cannot_be_approved(server, review_store):
    plan = _plan()
    plan["steps"][0]["ref"] = None
    status, _, record = request(server, "/api/update", method="POST", data={
        "plan": plan, "expected_revision": 0, "reviewed_steps": [],
    })
    assert status == 200
    assert record["revision"] == 1
    assert not record["validation"]["valid"]
    assert any(issue["code"] == "needs_ref" for issue in record["validation"]["issues"])
    status, _, error = request(server, "/api/approve", method="POST", data={
        "expected_revision": 1, "reviewed_steps": [0],
    })
    assert status == 400
    assert error["issues"]
    assert review_store.load()["status"] == "draft"


def test_stale_save_and_stale_approval_are_conflicts(server, review_store):
    status, _, record = request(server, "/api/update", method="POST", data={
        "plan": _plan(), "expected_revision": 0, "reviewed_steps": [0],
    })
    assert status == 200 and record["reviewed_steps"] == [0]
    assert request(server, "/api/update", method="POST", data={
        "plan": _plan(), "expected_revision": 0, "reviewed_steps": [],
    })[0] == 409
    assert request(server, "/api/approve", method="POST", data={
        "expected_revision": 0, "reviewed_steps": [0],
    })[0] == 409
    assert review_store.load()["revision"] == 1


def test_approval_only_changes_artifact_and_does_not_claim_or_execute(server, review_store, monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail("The HTTP service must never claim or execute a task.")
    monkeypatch.setattr(review_store, "claim", forbidden)
    assert request(server, "/api/approve", method="POST", data={
        "expected_revision": 0, "reviewed_steps": [],
    })[0] == 400
    status, _, result = request(server, "/api/approve", method="POST", data={
        "expected_revision": 0, "reviewed_steps": [0],
    })
    assert status == 200
    assert result["status"] == "approved"
    assert result["claim"] is None
    assert not (review_store.path / ".review-claimed.json").exists()


def test_external_binding_disallowed_before_socket_creation(review_store):
    for host in ("0.0.0.0", "192.168.1.1", "::", "example.com"):
        with pytest.raises(ValueError, match="bind"):
            ReviewServer(review_store, host=host, port=0)


def test_server_close_is_idempotent_without_start(review_store):
    server = ReviewServer(review_store, port=0)
    server.close()
    server.close()
    with pytest.raises(RuntimeError, match="closed"):
        server.start_background()


def test_foreground_wrapper_signals_ready_and_stops_without_execution(review_store):
    stop = threading.Event()
    urls = []
    def ready(url):
        urls.append(url)
        stop.set()
    serve(review_store, port=0, ready_callback=ready, stop_event=stop)
    assert len(urls) == 1 and "#token=" in urls[0]
