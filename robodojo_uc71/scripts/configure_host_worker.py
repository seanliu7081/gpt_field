#!/usr/bin/env python3
"""Transfer this app's OAuth store to its root parent and test worker isolation.

Run with sudo after browser sign-in. No service, simulator or model is started.
No other credentials, workspace trees or system configuration are modified.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "src"))
from robodojo_uc71.host_runtime import CREDENTIAL_STORE, verify_conda_record
from robodojo_uc71.worker_isolation import popen_identity, prepare_access


def validate_store_tree(store):
    if store.resolve(strict=True) != store or not store.is_dir():
        raise ValueError("canonical_app_credential_store_required")
    paths = [store, *store.rglob("*")]
    for path in paths:
        info = path.lstat()
        if (not (stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode))
                or (stat.S_ISREG(info.st_mode) and info.st_nlink != 1)):
            raise ValueError("plain_single_link_credential_files_required")
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-record", type=Path,
        default=BASE/"validation/conda_host_20261030/runtime_001/install_status.json")
    parser.add_argument("--output", type=Path,
        default=BASE/"runtime/host_fix_20261030/root_setup_001")
    parser.add_argument("--run-tests", action="store_true")
    args = parser.parse_args()
    if os.getuid() != 0 or os.geteuid() != 0:
        parser.error("sudo is required for the root parent and separate worker UID")
    verify_conda_record(sys.executable, args.runtime_record)
    paths = validate_store_tree(CREDENTIAL_STORE)
    owner = CREDENTIAL_STORE.stat()
    # Check sign-in as the current owner before transferring the app's store.
    probe = """import json,sys
from gpt_field.chatgpt_auth import SubscriptionCredentials
s=SubscriptionCredentials(sys.argv[1]).status()
print(json.dumps({'connected':s['connected'],'plan_use_granted':any(
 p['active'] and p['plan_use_granted'] for p in s['profiles'])}))
"""
    env = {"PATH":str(Path(sys.executable).parent)+":/usr/bin:/bin",
           "PYTHONNOUSERSITE":"1", "PYTHONDONTWRITEBYTECODE":"1"}
    status = json.loads(subprocess.check_output([sys.executable,"-I","-B","-c",probe,str(CREDENTIAL_STORE)],
        user=owner.st_uid, group=owner.st_gid, extra_groups=[], env=env, text=True, timeout=30))
    if not status["connected"] or not status["plan_use_granted"]:
        parser.error("complete browser sign-in and grant ChatGPT plan use first")
    output = args.output.absolute()
    if (output.exists() or output.parent.resolve(strict=True) != output.parent
            or not output.is_relative_to(BASE/"runtime")):
        parser.error("a fresh task-local runtime output directory is required")
    output.mkdir(mode=0o755)
    for path in paths:
        fd = os.open(path, os.O_RDONLY|os.O_NOFOLLOW)
        try:
            info = os.fstat(fd)
            if not stat.S_ISDIR(info.st_mode) and (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1):
                raise ValueError("credential_file_changed_during_setup")
            os.fchown(fd, 0, 0)
            os.fchmod(fd, 0o700 if stat.S_ISDIR(info.st_mode) else 0o600)
        finally:
            os.close(fd)
    layout = prepare_access(output/"probe", output/"runtime")
    worker = """import json,sys
from pathlib import Path
from robodojo_uc71.host_runtime import CREDENTIAL_STORE
from robodojo_uc71.worker_isolation import enforce_child_boundary
proof=enforce_child_boundary(credential_store=CREDENTIAL_STORE,parent_pid=int(sys.argv[1]))
import gpt_field,supervisor,importlib.metadata
assert importlib.metadata.version('isaacsim')=='5.1.0.0'
assert importlib.metadata.version('torch')=='2.7.0+cu128'
target=Path(sys.argv[2])/'worker_probe.json'
target.write_text(json.dumps({'worker_write_passed':True}))
proof['interpreter_and_package_metadata_readable']=True
proof['private_worker_write_passed']=target.is_file()
print(json.dumps(proof))
"""
    result = {"schema":"robodojo-current-host-worker-setup-1", "subscription":status,
        "credential_store":str(CREDENTIAL_STORE), "credential_owner_uid":0,
        "layout":layout, "simulator_starts":0, "model_calls":0, "benchmark_actions":0,
        "full_test_authorized":False}
    try:
        result["worker_isolation"] = json.loads(subprocess.check_output(
            [sys.executable,"-I","-B","-c",worker,str(os.getpid()),layout["worker_output"]],
            **popen_identity(),env=env,text=True,timeout=30))
        if args.run_tests:
            with tempfile.TemporaryDirectory(prefix="robodojo-root-tests-") as temporary:
                with (output/"pytest.log").open("x") as log:
                    completed = subprocess.run([sys.executable,"-B","-m","pytest","-q","-o","pythonpath=",
                        "-p","no:cacheprovider","--basetemp",str(Path(temporary)/"tests"),
                        "--junitxml",str(output/"pytest.xml"),str(BASE/"tests")],
                        env=env,cwd=BASE.parent,stdout=log,stderr=subprocess.STDOUT,timeout=300)
            result["pytest_returncode"] = completed.returncode
            if completed.returncode:
                raise RuntimeError("offline_tests_failed_see_pytest_log")
        result["passed"] = True
    except Exception as error:
        result["passed"] = False
        result["error_type"] = type(error).__name__
        raise
    finally:
        (output/"result.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"passed":True,"record":str(output/"result.json"),"model_calls":0}))


if __name__ == "__main__":
    main()
