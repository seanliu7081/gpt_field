#!/usr/bin/env python3
"""Share selected completed rollout videos through a private, temporary HTTPS link.

Usage: mobile_media.py start|status|stop
       mobile_media.py publish|link RUN_ID
Links require the secret query token. Restarting the service changes the URL and
revokes old tokens. Only curated copies in .private/mobile_media/public are served.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import secrets
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z")
TUNNEL_URL = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com\b")
TERMINAL = {"success", "completed", "complete", "failed", "failure", "error",
            "subscription_error", "user_stop", "stopped", "interrupted", "cancelled", "canceled"}


def checked_path(path: Path) -> Path:
    """Reject symlinks at every existing component, including the final file."""
    for part in (path, *path.parents):
        if part.is_symlink():
            raise ValueError(f"Symlinks are not allowed: {part}")
    return path


def run(command: list[str], **kwargs) -> subprocess.CompletedProcess:
    result = subprocess.run(command, capture_output=True, text=True, timeout=120, **kwargs)
    if result.returncode:
        # Do not echo commands, config contents or logs that could carry secrets.
        raise RuntimeError(f"{Path(command[0]).name} failed (exit {result.returncode})")
    return result


class MobileMedia:
    def __init__(self, root: Path, port: int = 17865, metrics_port: int = 17867):
        self.root = root.absolute()
        self.private = self.root / ".private" / "mobile_media"
        self.public = self.private / "public"
        self.config = self.private / "supervisord.conf"
        self.port, self.metrics_port = port, metrics_port

    def prepare(self):
        for path in (self.root / ".private", self.private, self.public):
            checked_path(path).mkdir(mode=0o700, exist_ok=True)
            path.chmod(0o700)
        for child in self.private.iterdir():
            checked_path(child)

    def write(self, name: str, value: str):
        path = checked_path(self.private / name)
        path.write_text(value)
        path.chmod(0o600)

    @staticmethod
    def validate_id(run_id: str):
        if not RUN_ID.fullmatch(run_id):
            raise ValueError("Run ID must contain only letters, digits, underscores and hyphens")

    def target(self, run_id: str) -> Path:
        self.validate_id(run_id)
        return checked_path(self.public / f"{run_id}.mp4")

    def source(self, run_id: str) -> Path:
        self.validate_id(run_id)
        directory = checked_path(self.root / "runs" / run_id)
        summary = checked_path(directory / "summary.json")
        if not summary.is_file():
            raise ValueError("Run needs a terminal summary.json before its video can be shared")
        data = json.loads(summary.read_text())
        if str(data.get("status", "")).lower() not in TERMINAL:
            raise ValueError("Run summary does not identify a terminal state")
        state = checked_path(directory / "run_state.json")
        if state.exists():
            current = json.loads(state.read_text())
            if str(current.get("status", "")).lower() not in TERMINAL:
                raise ValueError("Run state does not identify a terminal state")
        source = checked_path(directory / "episode.mp4")
        if not source.is_file():
            raise ValueError("Run has no completed episode.mp4")
        return source

    def publish(self, run_id: str) -> Path:
        source, target = self.source(run_id), self.target(run_id)
        before = source.stat()
        probe = json.loads(run(["ffprobe", "-v", "error", "-show_streams",
                                "-show_format", "-of", "json", str(source)]).stdout)
        streams = probe.get("streams", [])
        video = [s for s in streams if s.get("codec_type") == "video"]
        if (len(video) != 1 or video[0].get("codec_name") != "h264"
                or video[0].get("pix_fmt") != "yuv420p"
                or float(probe.get("format", {}).get("duration", 0)) <= 0):
            raise ValueError("Video must be a complete H.264 / yuv420p MP4 for iPad playback")
        if any(s.get("codec_type") == "audio" and s.get("codec_name") != "aac" for s in streams):
            raise ValueError("Audio must be AAC for iPad playback")
        fd, temporary = tempfile.mkstemp(prefix=".export-", suffix=".mp4", dir=self.private)
        os.close(fd)
        try:
            run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(source),
                 "-map", "0:v:0", "-map", "0:a?", "-c", "copy", "-movflags",
                 "+faststart", "-map_metadata", "-1", temporary])
            after = source.stat()
            fingerprint = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns)
            if fingerprint(before) != fingerprint(after):
                raise ValueError("Source changed during export; no video was published")
            if not Path(temporary).stat().st_size:
                raise ValueError("Empty export; no video was published")
            os.replace(temporary, target)
        finally:
            Path(temporary).unlink(missing_ok=True)
        return target

    def ctl(self, action: str) -> subprocess.CompletedProcess:
        return subprocess.run(["supervisorctl", "-c", str(self.config), action],
                              capture_output=True, text=True, timeout=15)

    def states(self) -> dict[str, str]:
        if not self.config.exists():
            return {}
        result = self.ctl("status")
        return {parts[0]: parts[1] for line in result.stdout.splitlines()
                if len(parts := line.split()) >= 2 and parts[0] in {"media", "tunnel"}}

    def origin_ready(self, host: str) -> bool:
        """Check forwarded-host routing, authentication and actual byte serving."""
        local = f"http://127.0.0.1:{self.port}"
        try:
            try:
                with urllib.request.urlopen(urllib.request.Request(local + "/", headers={"Host": host}), timeout=2):
                    return False
            except urllib.error.HTTPError as error:
                if error.code != 401:
                    return False
                error.close()
            token = (self.private / "token").read_text().strip()
            videos = sorted(self.public.glob("*.mp4"))
            path = checked_path(videos[0]).name if videos else "__readiness_missing__"
            request = urllib.request.Request(local + f"/{path}?token={token}",
                                             headers={"Host": host, "Range": "bytes=0-0"})
            try:
                with urllib.request.urlopen(request, timeout=2) as response:
                    return (bool(videos) and response.status == 206
                            and response.headers.get_content_type() == "video/mp4"
                            and len(response.read(2)) == 1)
            except urllib.error.HTTPError as error:
                expected = not videos and error.code == 404
                error.close()
                return expected
        except (OSError, ValueError, urllib.error.URLError):
            return False

    def base_url(self) -> str | None:
        # A fresh log is created before every supervisor start. An old URL must
        # never be returned after shutdown or while a replacement tunnel starts.
        if self.states() != {"media": "RUNNING", "tunnel": "RUNNING"}:
            return None
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{self.metrics_port}/ready", timeout=2) as response:
                if response.status != 200:
                    return None
            matches = TUNNEL_URL.findall((self.private / "tunnel.log").read_text())
            return matches[-1] if matches and self.origin_ready(matches[-1][8:]) else None
        except (OSError, urllib.error.URLError):
            return None

    def status(self) -> dict:
        states = self.states()
        return {"processes": states, "ready": bool(url := self.base_url()),
                "base_url": url, "published": sorted(p.stem for p in self.public.glob("*.mp4"))}

    def link(self, run_id: str) -> str:
        target = self.target(run_id)
        if not target.is_file():
            raise ValueError("Publish this run first")
        if base := self.base_url():
            token = (self.private / "token").read_text().strip()
            if not re.fullmatch(r"[A-Za-z0-9_-]{43}", token):
                raise ValueError("Invalid token file; restart the service")
            return f"{base}/{run_id}.mp4?token={token}"
        return str(target)

    def configurations(self, token: str) -> tuple[str, str]:
        caddy = """{
    admin off
    auto_https off
    persist_config off
}
http://:PORT {
    bind 127.0.0.1
    header Referrer-Policy no-referrer
    header Cache-Control "private, no-store"
    @authorized query token=TOKEN
    handle @authorized {
        root * ROOT
        file_server
    }
    handle {
        respond "Unauthorized" 401
    }
}
""".replace("PORT", str(self.port)).replace("TOKEN", token).replace("ROOT", json.dumps(str(self.public)))
        commands = {
            "media": [shutil.which("caddy"), "run", "--config", str(self.private / "Caddyfile"), "--adapter", "caddyfile"],
            "tunnel": [shutil.which("cloudflared"), "tunnel", "--no-autoupdate", "--protocol", "http2",
                       "--url", f"http://127.0.0.1:{self.port}", "--metrics",
                       f"127.0.0.1:{self.metrics_port}", "--logfile", str(self.private / "tunnel.log")],
        }
        supervisor = f"""[unix_http_server]
file={self.private}/supervisor.sock
chmod=0600
[supervisord]
logfile={self.private}/supervisord.log
pidfile={self.private}/supervisord.pid
childlogdir={self.private}
umask=0077
[rpcinterface:supervisor]
supervisor.rpcinterface_factory=supervisor.rpcinterface:make_main_rpcinterface
[supervisorctl]
serverurl=unix://{self.private}/supervisor.sock
"""
        for name, command in commands.items():
            supervisor += f"""
[program:{name}]
command={shlex.join(command).replace('%', '%%')}
directory={self.private}
autostart=true
autorestart=true
startsecs=2
startretries=3
stopasgroup=true
killasgroup=true
stdout_logfile=/dev/stdout
stdout_logfile_maxbytes=0
redirect_stderr=true
"""
        return caddy, supervisor

    def start(self, timeout: float = 35) -> dict:
        states = self.states()
        if states:
            if set(states.values()) != {"RUNNING"}:
                raise RuntimeError("Private media supervisor exists but is unhealthy; use stop, then start")
            return self.status()
        for command in ("caddy", "cloudflared", "supervisord", "supervisorctl"):
            if not shutil.which(command):
                raise RuntimeError(f"Missing required executable: {command}")
        for port in (self.port, self.metrics_port):
            with socket.socket() as probe:
                probe.bind(("127.0.0.1", port))
        token = secrets.token_urlsafe(32)
        caddy, supervisor = self.configurations(token)
        self.write("token", token)
        self.write("Caddyfile", caddy)
        self.write("supervisord.conf", supervisor)
        self.write("tunnel.log", "")
        run(["caddy", "validate", "--config", str(self.private / "Caddyfile"), "--adapter", "caddyfile"])
        # Supervisor daemonizes itself. Detach all inherited streams so the IDE
        # command cannot hang on an open pipe and no token reaches terminal logs.
        result = subprocess.run(["supervisord", "-c", str(self.config)],
                                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL, timeout=15, start_new_session=True)
        if result.returncode:
            raise RuntimeError("Private supervisord failed to start")
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.base_url():
                return self.status()
            time.sleep(0.5)
        return self.status()

    def stop(self) -> dict:
        if self.states():
            result = self.ctl("shutdown")
            if result.returncode:
                raise RuntimeError("Private supervisor shutdown failed")
            deadline = time.monotonic() + 15
            while self.states() and time.monotonic() < deadline:
                time.sleep(0.2)
            if self.states():
                raise RuntimeError("Private supervisor has not stopped yet")
        self.write("tunnel.log", "")
        return {"stopped": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("start", "status", "stop", "publish", "link"))
    parser.add_argument("run_id", nargs="?")
    args = parser.parse_args()
    if (args.command in {"publish", "link"}) != bool(args.run_id):
        parser.error("Only publish and link require RUN_ID")
    media = MobileMedia(Path(__file__).resolve().parents[1])
    try:
        media.prepare()
        with checked_path(media.private / "lock").open("a") as lock:
            os.chmod(lock.name, 0o600)
            fcntl.flock(lock, fcntl.LOCK_EX)
            if args.command == "publish":
                media.publish(args.run_id)
                print(media.link(args.run_id))
            elif args.command == "link":
                print(media.link(args.run_id))
            else:
                print(json.dumps(getattr(media, args.command)(), indent=2))
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
        print(f"mobile_media: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
