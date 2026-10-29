"""Private process supervision tested solely with a harmless local dummy child."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

import pytest

HELPERS = Path(__file__).resolve().parents[1] / "runs/visual_field_v32_zero_shot_001"


def helper():
    spec = importlib.util.spec_from_file_location("v32_private_supervisor", HELPERS / "private_supervisor.py")
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


def test_v32_plan_preserves_official_input_and_unlimited_original_only_scope():
    import hashlib
    plan = json.loads((HELPERS / "plan.json").read_text())
    prior = json.loads((HELPERS.parent / "visual_field_v31_zero_shot_001/plan.json").read_text())
    assert plan["method_version"] == "v32" and plan["shots"] == 0
    assert len(plan["scenes"]) == 1
    scene, old = plan["scenes"][0], prior["scenes"][0]
    for key in ("id", "task", "suite", "task_id", "seed", "episode_index", "task_prompt", "bddl", "initial_state", "official_inputs", "expected_initial_sha256"):
        assert scene[key] == old[key]
    assert Path(scene["expected_initial"]).read_bytes() == Path(old["expected_initial"]).read_bytes()
    assert hashlib.sha256(Path(scene["expected_initial"]).read_bytes()).hexdigest() == scene["expected_initial_sha256"]
    argv = plan["command_argv"]
    assert argv[:3] == ["env", "-u", "OPENAI_API_KEY"]
    assert argv[argv.index("--method-version")+1] == "v32"
    assert argv[argv.index("--output")+1] == scene["output"]
    assert argv[argv.index("--expected-initial")+1] == scene["expected_initial"]
    assert "--demo" not in argv
    assert plan["model"] == "gpt-6-astra" and plan["effort"] == "medium"
    assert plan["billing"] == "chatgpt_subscription" and plan["api_key_fallback"] is False
    assert plan["max_episode_steps"] is None and plan["max_episode_wall_time_s"] is None
    assert plan["process_supervision"]["autorestart"] is False
    assert plan["process_supervision"]["startretries"] == 0
    assert plan["process_supervision"]["queue"] is False
    assert set(plan['new_runtime_modules']) == {'visual_reference_candidates_v32.py', 'visual_goal_replay_v32.py', 'visual_translation_guard_v32.py', 'visual_field_policy_v32.py', 'visual_objective_v32.py', 'visual_field_subscription_v32.py', 'visual_reference_track_v32.py', 'visual_monitor_v32.py'}
    assert 'visual_field_v31.py' in plan['inherited_v31_runtime_modules']


def test_changed_source_seal_is_refused_without_starting_any_process(tmp_path):
    import hashlib
    root = tmp_path / "project"
    directory = root / "runs" / "prepared"
    directory.mkdir(parents=True)
    source = root / "src/gpt_field/fake.py"
    source.parent.mkdir(parents=True)
    source.write_text("ORIGINAL = True\n")
    plan = {"scenes": [{"output": str(directory / "never_created")}]}
    (directory / "plan.json").write_text(json.dumps(plan))
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    seal = {"checks_passed": True, "source_freeze": "sealed",
        "plan_sha256": digest(directory / "plan.json"),
        "source_sha256": {"src/gpt_field/fake.py": digest(source)}, "helper_sha256": {}}
    (directory / "preflight.json").write_text(json.dumps(seal))
    for name in ("launch.py", "foreground_trial.py", "private_supervisor.py"):
        (directory / name).write_bytes((HELPERS / name).read_bytes())
    source.write_text("CHANGED = True\n")
    result = subprocess.run([sys.executable, str(directory / "launch.py")], capture_output=True, text=True, timeout=10)
    assert result.returncode != 0 and "Source differs from the seal" in result.stderr
    assert not (directory / "supervisor").exists()
    assert not (directory / "launch.json").exists()


@pytest.mark.parametrize('changed', ['frozen_initial_observation', 'official_bddl', 'official_initial_state', 'missing_seal_check'])
def test_sealed_input_is_rechecked_immediately_before_any_supervisor_launch(tmp_path, changed):
    import hashlib
    root = tmp_path / 'project'
    directory = root / 'runs' / 'prepared'
    directory.mkdir(parents=True)
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    files = {}
    for name in ('frozen_initial_observation', 'official_bddl', 'official_initial_state'):
        path = root / (name+'.txt')
        path.write_text('original '+name)
        files[name] = {'path': str(path), 'sha256': digest(path)}
    scene = {'output': str(directory / 'never_created'),
        'expected_initial': files['frozen_initial_observation']['path'],
        'expected_initial_sha256': files['frozen_initial_observation']['sha256'],
        'official_inputs': {key: value for key, value in files.items() if key != 'frozen_initial_observation'}}
    (directory / 'plan.json').write_text(json.dumps({'scenes': [scene]}))
    seal = {'checks_passed': True, 'source_freeze': 'sealed',
        'plan_sha256': digest(directory / 'plan.json'), 'source_sha256': {}, 'helper_sha256': {},
        'checks': [{'name': key, 'passed': True, **value} for key, value in files.items()]}
    if changed == 'missing_seal_check':
        seal['checks'].pop()
    else:
        Path(files[changed]['path']).write_text('changed after seal')
    (directory / 'preflight.json').write_text(json.dumps(seal))
    for name in ('launch.py', 'foreground_trial.py', 'private_supervisor.py'):
        (directory / name).write_bytes((HELPERS / name).read_bytes())
    result = subprocess.run([sys.executable, str(directory / 'launch.py')],
        capture_output=True, text=True, timeout=10)
    assert result.returncode != 0
    assert ('Required input is not bound to the seal' if changed == 'missing_seal_check'
            else 'Input differs from the seal') in result.stderr
    assert not (directory / 'supervisor').exists()
    assert not (directory / 'launch.json').exists()


def test_unchanged_complete_input_seal_verifies_without_launch(tmp_path):
    import hashlib
    root = tmp_path / 'project'
    directory = root / 'runs' / 'prepared'
    directory.mkdir(parents=True)
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    files = {}
    for name in ('frozen_initial_observation', 'official_bddl', 'official_initial_state'):
        path = root / (name+'.txt'); path.write_text('original '+name)
        files[name] = {'path': str(path), 'sha256': digest(path)}
    scene = {'output': str(directory / 'never_created'),
        'expected_initial': files['frozen_initial_observation']['path'],
        'expected_initial_sha256': files['frozen_initial_observation']['sha256'],
        'official_inputs': {key: value for key, value in files.items() if key != 'frozen_initial_observation'}}
    plan = {'scenes': [scene]}
    (directory / 'plan.json').write_text(json.dumps(plan))
    (directory / 'foreground_trial.py').write_bytes((HELPERS / 'foreground_trial.py').read_bytes())
    seal = {'checks_passed': True, 'source_freeze': 'sealed',
        'plan_sha256': digest(directory / 'plan.json'), 'source_sha256': {},
        'helper_sha256': {'foreground_trial.py': digest(directory / 'foreground_trial.py')},
        'checks': [{'name': key, 'passed': True, **value} for key, value in files.items()]}
    (directory / 'preflight.json').write_text(json.dumps(seal))
    spec = importlib.util.spec_from_file_location('isolated_v32_foreground', directory / 'foreground_trial.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    assert module.verify() == plan
    assert not (directory / 'supervisor').exists()
    assert not (directory / 'launch.json').exists()
