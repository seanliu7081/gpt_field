"""Safety and lifecycle checks for the isolated iPad rollout publisher."""
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest

SPEC = importlib.util.spec_from_file_location("mobile_media", Path(__file__).parents[1] / "scripts" / "mobile_media.py")
mobile = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mobile)


@pytest.fixture
def media(tmp_path):
    service = mobile.MobileMedia(tmp_path)
    service.prepare()
    return service


def source(media, status="success"):
    directory = media.root / "runs" / "example_001"
    directory.mkdir(parents=True)
    (directory / "summary.json").write_text(json.dumps({"status": status}))
    (directory / "run_state.json").write_text(json.dumps({"status": status}))
    video = directory / "episode.mp4"
    video.write_bytes(b"original video")
    return video


def fake_tools(monkeypatch, *, fail=False, change=None, codec="h264", pix_fmt="yuv420p"):
    def execute(command, **kwargs):
        if command[0] == "ffprobe":
            data = {"streams": [{"codec_type": "video", "codec_name": codec, "pix_fmt": pix_fmt}],
                    "format": {"duration": "1.0"}}
            return subprocess.CompletedProcess(command, 0, json.dumps(data), "")
        assert command[0] == "ffmpeg"
        assert command[command.index("-c") + 1] == "copy"
        assert "+faststart" in command
        Path(command[-1]).write_bytes(b"faststart video")
        if change:
            change.write_bytes(b"video still being generated")
        if fail:
            raise RuntimeError("ffmpeg failed")
        return subprocess.CompletedProcess(command, 0, "", "")
    monkeypatch.setattr(mobile, "run", execute)


@pytest.mark.parametrize("value", ["../outside", "a/b", "a.mp4", "", "/tmp/foo", "a?token=x", "a\\b", ".hidden", "a" * 129])
def test_invalid_run_ids(media, value):
    with pytest.raises(ValueError):
        media.source(value)
    with pytest.raises(ValueError):
        media.target(value)


@pytest.mark.parametrize("part", ["runs", "example_001", "episode.mp4", "summary.json", "run_state.json"])
def test_source_symlinks_rejected(media, tmp_path, part):
    video = source(media)
    path = (media.root / "runs" if part == "runs" else
            video.parent if part == "example_001" else video.parent / part)
    replacement = tmp_path / "moved"
    path.rename(replacement)
    path.symlink_to(replacement, target_is_directory=replacement.is_dir())
    with pytest.raises(ValueError, match="Symlinks"):
        media.source("example_001")


def test_public_symlink_rejected(media, tmp_path):
    (media.public / "example_001.mp4").symlink_to(tmp_path / "outside")
    with pytest.raises(ValueError, match="Symlinks"):
        media.target("example_001")


def test_private_directory_permissions_and_symlink(media, tmp_path):
    assert media.private.stat().st_mode & 0o777 == 0o700
    assert media.public.stat().st_mode & 0o777 == 0o700
    media.public.rmdir()
    media.public.symlink_to(tmp_path)
    with pytest.raises(ValueError, match="Symlinks"):
        media.prepare()


@pytest.mark.parametrize("status", ["running", "starting", "waiting", "unknown", ""])
def test_active_and_unknown_runs_rejected(media, status):
    source(media, status)
    with pytest.raises(ValueError, match="terminal"):
        media.source("example_001")


def test_active_state_overrides_old_terminal_summary(media):
    video = source(media)
    (video.parent / "run_state.json").write_text('{"status":"running"}')
    with pytest.raises(ValueError, match="terminal"):
        media.source("example_001")


def test_summary_required(media):
    video = source(media)
    (video.parent / "summary.json").unlink()
    with pytest.raises(ValueError, match="summary"):
        media.source("example_001")


def test_export_preserves_source_and_publishes_atomically(media, monkeypatch):
    video = source(media)
    before = video.stat()
    fake_tools(monkeypatch)
    output = media.publish("example_001")
    assert output.read_bytes() == b"faststart video"
    assert video.read_bytes() == b"original video"
    assert video.stat().st_mtime_ns == before.st_mtime_ns
    assert output.stat().st_mode & 0o777 == 0o600
    assert not list(media.private.glob(".export-*"))


@pytest.mark.parametrize("existing", [False, True])
def test_failed_export_never_publishes_partial(media, monkeypatch, existing):
    source(media)
    target = media.target("example_001")
    if existing:
        target.write_bytes(b"older complete copy")
    fake_tools(monkeypatch, fail=True)
    with pytest.raises(RuntimeError):
        media.publish("example_001")
    assert target.read_bytes() == b"older complete copy" if existing else not target.exists()
    assert not list(media.private.glob(".export-*"))


def test_source_change_aborts_publication(media, monkeypatch):
    video = source(media)
    fake_tools(monkeypatch, change=video)
    with pytest.raises(ValueError, match="changed"):
        media.publish("example_001")
    assert not media.target("example_001").exists()


@pytest.mark.parametrize("codec,pix_fmt", [("hevc", "yuv420p"), ("h264", "yuv444p")])
def test_incompatible_video_rejected(media, monkeypatch, codec, pix_fmt):
    source(media)
    fake_tools(monkeypatch, codec=codec, pix_fmt=pix_fmt)
    with pytest.raises(ValueError, match="iPad"):
        media.publish("example_001")


def test_configuration_isolated_and_authenticated(media, monkeypatch):
    monkeypatch.setattr(mobile.shutil, "which", lambda name: f"/bin/{name}")
    caddy, supervisor = media.configurations("secret")
    assert "http://:17865" in caddy
    assert "bind 127.0.0.1" in caddy
    assert "admin off" in caddy and "auto_https off" in caddy
    assert "persist_config off" in caddy
    assert "@authorized query token=secret" in caddy
    assert 'respond "Unauthorized" 401' in caddy
    assert str(media.public) in caddy
    assert str(media.root / "runs") not in caddy
    assert "browse" not in caddy
    assert "Referrer-Policy no-referrer" in caddy
    assert 'Cache-Control "private, no-store"' in caddy
    assert "stdout_logfile=/dev/stdout" in supervisor
    assert "--protocol http2" in supervisor
    assert f"serverurl=unix://{media.private}/supervisor.sock" in supervisor
    assert "--url http://127.0.0.1:17865" in supervisor
    assert "--metrics 127.0.0.1:17867" in supervisor
    assert "/etc/supervisor" not in supervisor and "secret" not in supervisor


def test_link_requires_running_ready_tunnel(media, monkeypatch):
    media.target("example_001").write_bytes(b"video")
    media.write("token", "A" * 43)
    monkeypatch.setattr(media, "base_url", lambda: None)
    assert media.link("example_001") == str(media.target("example_001"))
    monkeypatch.setattr(media, "base_url", lambda: "https://new-link.trycloudflare.com")
    assert media.link("example_001") == "https://new-link.trycloudflare.com/example_001.mp4?token=" + "A" * 43


def test_status_never_contains_token(media, monkeypatch):
    media.write("token", "A" * 43)
    monkeypatch.setattr(media, "states", lambda: {"media": "RUNNING", "tunnel": "RUNNING"})
    monkeypatch.setattr(media, "base_url", lambda: "https://new-link.trycloudflare.com")
    assert "A" * 43 not in json.dumps(media.status())


def test_stale_tunnel_url_ignored_when_stopped(media, monkeypatch):
    media.write("tunnel.log", "https://stale-link.trycloudflare.com")
    monkeypatch.setattr(media, "states", lambda: {})
    assert media.base_url() is None


def test_tunnel_must_be_ready_and_latest_url_wins(media, monkeypatch):
    monkeypatch.setattr(media, "origin_ready", lambda host: host == "new-link.trycloudflare.com")
    media.write("tunnel.log", "https://old-link.trycloudflare.com\nhttps://new-link.trycloudflare.com")
    monkeypatch.setattr(media, "states", lambda: {"media": "RUNNING", "tunnel": "RUNNING"})
    class Response:
        status = 200
        def __enter__(self): return self
        def __exit__(self, *args): pass
    monkeypatch.setattr(mobile.urllib.request, "urlopen", lambda *args, **kwargs: Response())
    assert media.base_url() == "https://new-link.trycloudflare.com"
    Response.status = 503
    assert media.base_url() is None


def test_start_clears_stale_url_rotates_token_and_detaches_streams(media, monkeypatch):
    media.port = media.metrics_port = 0
    media.write("tunnel.log", "https://stale-link.trycloudflare.com")
    media.write("token", "old-token")
    monkeypatch.setattr(media, "states", lambda: {})
    monkeypatch.setattr(mobile.shutil, "which", lambda name: f"/bin/{name}")
    monkeypatch.setattr(mobile, "run", lambda *args, **kwargs: None)
    monkeypatch.setattr(media, "base_url", lambda: None)
    calls = []
    def spawn(command, **kwargs):
        calls.append((command, kwargs))
        assert (media.private / "tunnel.log").read_text() == ""
        assert (media.private / "token").read_text() != "old-token"
        return subprocess.CompletedProcess(command, 0)
    monkeypatch.setattr(mobile.subprocess, "run", spawn)
    media.start(timeout=0)
    assert len(calls) == 1
    assert calls[0][1]["stdout"] == subprocess.DEVNULL
    assert calls[0][1]["stderr"] == subprocess.DEVNULL
    assert calls[0][1]["start_new_session"] is True
    assert (media.private / "token").stat().st_mode & 0o777 == 0o600


def test_start_idempotent(media, monkeypatch):
    monkeypatch.setattr(media, "states", lambda: {"media": "RUNNING", "tunnel": "RUNNING"})
    monkeypatch.setattr(media, "base_url", lambda: "https://existing-link.trycloudflare.com")
    monkeypatch.setattr(mobile.subprocess, "run", lambda *args, **kwargs: pytest.fail("Unexpected process launch"))
    assert media.start()["ready"]


def test_stop_only_own_supervisor_and_clears_log(media, monkeypatch):
    states = iter([{"media": "RUNNING", "tunnel": "RUNNING"}, {}, {}])
    monkeypatch.setattr(media, "states", lambda: next(states))
    actions = []
    monkeypatch.setattr(media, "ctl", lambda action: actions.append(action) or subprocess.CompletedProcess([], 0))
    media.write("tunnel.log", "https://stale-link.trycloudflare.com")
    assert media.stop() == {"stopped": True}
    assert actions == ["shutdown"]
    assert (media.private / "tunnel.log").read_text() == ""


def test_tunnel_with_unhealthy_origin_is_not_ready(media, monkeypatch):
    media.write("tunnel.log", "https://current-link.trycloudflare.com")
    monkeypatch.setattr(media, "states", lambda: {"media": "RUNNING", "tunnel": "RUNNING"})
    monkeypatch.setattr(media, "origin_ready", lambda host: False)
    class Response:
        status = 200
        def __enter__(self): return self
        def __exit__(self, *args): pass
    monkeypatch.setattr(mobile.urllib.request, "urlopen", lambda *args, **kwargs: Response())
    assert media.base_url() is None


@pytest.mark.skipif(not mobile.shutil.which("caddy"), reason="Caddy is not installed")
def test_real_caddy_forwarded_host_auth_range_and_no_browsing(media):
    # A scoped throwaway process verifies real HTTP behavior, without touching
    # the running private supervisor or its ports, config, token or exports.
    with mobile.socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        media.port = listener.getsockname()[1]
    token = "A" * 43
    media.write("token", token)
    data = b"0123456789example MP4 bytes"
    media.target("example_001").write_bytes(data)
    media.write("private-secret.txt", "must stay private")
    caddy, _ = media.configurations(token)
    media.write("Caddyfile", caddy)
    process = subprocess.Popen(["caddy", "run", "--config", str(media.private / "Caddyfile"),
                                "--adapter", "caddyfile"], stdin=subprocess.DEVNULL,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    host = "forwarded-host.trycloudflare.com"
    def request(path, **headers):
        req = mobile.urllib.request.Request(f"http://127.0.0.1:{media.port}{path}",
                                            headers={"Host": host, **headers})
        try:
            return mobile.urllib.request.urlopen(req, timeout=2)
        except mobile.urllib.error.HTTPError as error:
            return error
    try:
        deadline = mobile.time.monotonic() + 5
        while not media.origin_ready(host) and mobile.time.monotonic() < deadline:
            assert process.poll() is None, "Temporary Caddy exited during startup"
            mobile.time.sleep(0.05)
        assert media.origin_ready(host)
        with request("/example_001.mp4") as response:
            assert response.status == 401
        with request("/example_001.mp4?token=wrong") as response:
            assert response.status == 401
        with request("/example_001.mp4?token=" + token, Range="bytes=2-5") as response:
            assert response.status == 206
            assert response.read() == data[2:6]
            assert response.headers.get_content_type() == "video/mp4"
            assert response.headers["Content-Range"] == f"bytes 2-5/{len(data)}"
            assert response.headers["Referrer-Policy"] == "no-referrer"
            assert response.headers["Cache-Control"] == "private, no-store"
        for path in ("/", "/private-secret.txt", "/../private-secret.txt", "/runs/example_001/episode.mp4"):
            with request(path + "?token=" + token) as response:
                assert response.status == 404
        media.target("example_001").unlink()
        assert media.origin_ready(host), "Empty curated root should still be healthy"
        media.write("token", "B" * 43)
        assert not media.origin_ready(host), "Incorrect auth must fail readiness"
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
