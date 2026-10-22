"""Launch exactly one approved episode under a private supervisor (no restart)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approved-full-test", action="store_true")
    parser.add_argument("--name", default="drawer_bowl_001")
    args = parser.parse_args(argv)
    if not args.approved_full_test:
        parser.error("User approval is required before a full test; no process was launched")
    if not args.name or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_" for c in args.name):
        parser.error("Use a simple lowercase run name")
    destination = ROOT / "astranomy/runs" / args.name
    management = ROOT / "astranomy/launch" / args.name
    if destination.exists() or management.exists():
        parser.error("Choose a fresh name; existing episodes are never overwritten/resumed")
    supervisor = shutil.which("supervisord")
    if supervisor is None:
        parser.error("supervisord is unavailable")
    management.mkdir(parents=True)
    config = management / "supervisord.conf"
    command = f"{sys.executable} -u -m astranomy.run --run --approved-full-test --output {destination}"
    config.write_text(f'''[unix_http_server]
file={management}/control.sock
chmod=0600

[supervisord]
logfile={management}/supervisord.log
pidfile={management}/supervisord.pid
childlogdir={management}

[rpcinterface:supervisor]
supervisor.rpcinterface_factory=supervisor.rpcinterface:make_main_rpcinterface

[supervisorctl]
serverurl=unix://{management}/control.sock

[program:astranomy_drawer]
command={command}
directory={ROOT}
environment=PYTHONPATH="{ROOT}/src:{ROOT}",PYTHONDONTWRITEBYTECODE="1",PROC_NAME="astranomy_drawer"
autostart=true
autorestart=false
startsecs=0
startretries=0
stopsignal=INT
stopwaitsecs=240
stopasgroup=false
killasgroup=false
redirect_stderr=true
stdout_logfile={management}/runner.log
stdout_logfile_maxbytes=0
''')
    # This is a private, local-only supervisor, not an internet service. It does
    # not change the instance's Caddy, portal, or managed services.
    subprocess.run([supervisor, "-c", str(config)], check=True)
    print(json.dumps({"output": str(destination), "supervisor_config": str(config),
                      "graceful_stop": f"touch {destination}/operator_stop.request.json",
                      "shutdown_after_episode": f"supervisorctl -c {config} shutdown"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
