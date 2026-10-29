"""Private process supervision tested solely with a harmless local dummy child."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

import pytest

HELPERS = Path(__file__).resolve().parents[1] / "runs/visual_field_v29_zero_shot_001"


def helper():
    spec = importlib.util.spec_from_file_location("v29_private_supervisor", HELPERS / "private_supervisor.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def test_dummy_child_survives_launcher_exit_and_failed_exit_does_not_restart(tmp_path):
    module = helper()
    if not Path(module.SUPERVISORD).exists(): pytest.skip("Private supervisord binary unavailable")
    calls, complete = tmp_path / "calls.jsonl", tmp_path / "complete.json"
    child = tmp_path / "dummy_child.py"
    child.write_text("import json,os,time\nfrom pathlib import Path\n"
        f"with Path({str(calls)!r}).open('a') as f: f.write(json.dumps({{'pid':os.getpid()}})+'\\n')\n"
        "time.sleep(.4)\n" + f"Path({str(complete)!r}).write_text('finished')\n" + "raise SystemExit(7)\n")
    config = module.prepare(tmp_path / "private", [sys.executable, str(child)],
                            working_directory=tmp_path, stdout_log=tmp_path / "child.log")
    text = config.read_text()
    assert "[inet_http_server]" not in text and "unix://" in text
    assert "autorestart=false" in text and "startretries=0" in text and "startsecs=0" in text
    assert config.stat().st_mode & 0o777 == 0o600
    # The launcher is a separate short-lived process. It exits before the child;
    # the daemon and foreground dummy are independent of its exec-session life.
    launcher = subprocess.run([module.SUPERVISORD, "-c", str(config)], capture_output=True, text=True, timeout=20)
    assert launcher.returncode == 0, launcher.stderr
    try:
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            status = module.control(config, "status", "trial")
            if complete.exists() and "EXITED" in status.stdout: break
            time.sleep(.05)
        assert complete.read_text() == "finished"
        assert "EXITED" in status.stdout
        time.sleep(.3)
        assert len(calls.read_text().splitlines()) == 1
        pid = json.loads(calls.read_text().splitlines()[0])["pid"]
        assert not Path(f"/proc/{pid}").exists()
        assert "exit status 7; not expected" in (config.parent / "supervisord.log").read_text()
        with pytest.raises(RuntimeError, match="already exists"):
            module.prepare(config.parent, [sys.executable, str(child)],
                           working_directory=tmp_path, stdout_log=tmp_path / "child.log")
    finally:
        assert module.control(config, "shutdown").returncode == 0


def test_real_trial_helper_refuses_unsealed_plan_without_launching(tmp_path):
    trial = tmp_path / "runs" / "trial"
    trial.mkdir(parents=True)
    for name in ("launch.py", "foreground_trial.py", "private_supervisor.py"):
        (trial / name).write_bytes((HELPERS / name).read_bytes())
    (trial / "plan.json").write_text("{}")
    (trial / "preflight.json").write_text(json.dumps({"checks_passed": False, "source_freeze": "provisional"}))
    result = subprocess.run([sys.executable, str(trial / "launch.py")], capture_output=True, text=True, timeout=10)
    assert result.returncode != 0
    assert "Root must finish validation and seal before launch" in result.stderr
    assert not (trial / "supervisor").exists()
    assert not (trial / "launch.json").exists()
