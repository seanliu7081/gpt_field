"""Create a private supervisor for one reserved episode; never auto-start or approve."""
import argparse
import json
from pathlib import Path
import shlex


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--reservation-file", type=Path, required=True)
    parser.add_argument("--store-dir", type=Path,
                        default=Path("/workspace/gpt_field/.private/chatgpt_plan"))
    reset = parser.add_mutually_exclusive_group(required=True)
    reset.add_argument("--enroll-reset", action="store_true")
    reset.add_argument("--expected-initial-observation-sha256")
    args = parser.parse_args()
    root = args.root.resolve()
    reservation = args.reservation_file.resolve()
    data = json.loads(reservation.read_text())
    trial = data["reservation_id"]
    if not trial.startswith("trial_") or not trial[6:].isdigit():
        raise ValueError("invalid reservation id")
    launch = root / "launches" / trial
    launch.mkdir(parents=True, exist_ok=False)
    session = root / "evaluations" / trial
    runner = root / "baseline/scripts/evaluate_speedup_evidence.py"
    base = ["/usr/bin/env", "-u", "CUDA_VISIBLE_DEVICES",
            "PYTHONPATH=" + str(root / "baseline/src"), "MUJOCO_GL=egl", "PYOPENGL_PLATFORM=egl",
            "MUJOCO_EGL_DEVICE_ID=1", "OMP_NUM_THREADS=1", "OPENBLAS_NUM_THREADS=1",
            "/workspace/gpt_field/.venv/bin/python", "-u", str(runner)]
    prepare = base + ["prepare", "--session-dir", str(session), "--reservation-file", str(reservation),
                      "--store-dir", str(args.store_dir.resolve())]
    prepare += (["--enroll-reset"] if args.enroll_reset else
                ["--expected-initial-observation-sha256", args.expected_initial_observation_sha256])
    run = base + ["run", "--session-dir", str(session), "--store-dir", str(args.store_dir.resolve())]
    config = f"""[unix_http_server]
file={launch}/control.sock
chmod=0600
[supervisord]
logfile={launch}/supervisord.log
pidfile={launch}/supervisord.pid
childlogdir={launch}
[rpcinterface:supervisor]
supervisor.rpcinterface_factory=supervisor.rpcinterface:make_main_rpcinterface
[supervisorctl]
serverurl=unix://{launch}/control.sock
"""
    for name, command in (("prepare", prepare), ("run", run)):
        config += f"""
[program:{name}]
command={shlex.join(command)}
directory={root}/baseline
autostart=false
autorestart=false
startsecs=0
startretries=0
stopsignal=INT
stopwaitsecs=240
stopasgroup=false
killasgroup=false
redirect_stderr=true
stdout_logfile_maxbytes=0
stdout_logfile={launch}/{name}.log
"""
    (launch / "supervisord.conf").write_text(config)
    (launch / "commands.json").write_text(json.dumps({"prepare": prepare, "run": run,
                                                     "automatic_retry": False,
                                                     "review_required": True}, indent=2))
    print(launch / "supervisord.conf")


if __name__ == "__main__":
    main()
