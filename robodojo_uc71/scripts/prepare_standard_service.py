#!/usr/bin/env python3
"""Write a private, disabled supervisor configuration; never start a service."""
import argparse
import json
import os
from pathlib import Path
import shlex


BASE = Path(__file__).resolve().parents[1]


def prepare_service(output, *, python, spec, overlay, compatibility):
    output, python, spec = Path(output).absolute(), Path(python).absolute(), Path(spec).absolute()
    if os.getuid() != 0 or output.exists() or len(str(output / "control.sock")) > 100:
        raise ValueError("fresh_short_private_supervisor_path_required")
    for path in (python, spec, BASE / "scripts/run_standard_attempt.py"):
        if not path.is_file() or any(c in str(path) for c in "\n\r%\";"):
            raise ValueError("invalid_service_path")
    # Only the frozen, nonsecret config is group-readable. Control socket,
    # process logs and supervisor PID stay root-only under umask0077.
    output.mkdir(mode=0o710)
    os.chown(output, 0, 65534); os.chmod(output, 0o710)
    command = shlex.join([str(python), "-B", "-u", str(BASE / "scripts/run_standard_attempt.py"),
                          "--spec", str(spec), "--launch"])
    imports = os.pathsep.join(map(str, (BASE / "src", compatibility, overlay)))
    config = f"""[unix_http_server]
file={output}/control.sock
chmod=0600

[supervisord]
logfile={output}/supervisord.log
pidfile={output}/supervisord.pid
childlogdir={output}
umask=0077

[rpcinterface:supervisor]
supervisor.rpcinterface_factory=supervisor.rpcinterface:make_main_rpcinterface

[supervisorctl]
serverurl=unix://{output}/control.sock

[program:robodojo_standard]
command={command}
directory={BASE.parent}
environment=UC71_STANDARD_SUPERVISOR="1",PATH="{python.parent}:/usr/bin:/bin",PYTHONPATH="{imports}",PYTHONNOUSERSITE="1",PYTHONDONTWRITEBYTECODE="1"
autostart=false
autorestart=false
startretries=0
startsecs=0
stopsignal=TERM
stopwaitsecs=90
stopasgroup=true
killasgroup=true
redirect_stderr=true
stdout_logfile={output}/parent.log
stdout_logfile_maxbytes=0
"""
    path = output / "supervisord.conf"
    with path.open("x") as stream: stream.write(config)
    os.chown(path, 0, 65534); path.chmod(0o640)
    result = {"schema": "robodojo-private-standard-supervisor-1", "configuration": str(path),
        "autostart": False, "autorestart": False, "startretries": 0, "service_started": False,
        "simulator_starts": 0, "model_calls": 0, "full_test_authorized": False,
        "start_requires_existing_exact_human_approval": True}
    with (output / "preparation.json").open("x") as stream: json.dump(result, stream, indent=2)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output", "python", "spec", "overlay", "compatibility"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare_service(args.output, python=args.python, spec=args.spec,
        overlay=args.overlay, compatibility=args.compatibility)))
