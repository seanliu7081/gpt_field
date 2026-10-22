"""Launch exactly one of the user's three authorized fresh LIBERO-10 episodes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

PROJECT = Path("/workspace/gpt_field")
PREPARATION = PROJECT / "astranomy/libero10_high_001"
RUNTIME = PREPARATION / "runtime"
parser = argparse.ArgumentParser()
parser.add_argument("--task", choices=["libero_10_t2", "libero_10_t6", "libero_10_t8"], required=True)
parser.add_argument("--gpu", choices=["2", "3", "4"], required=True)
args = parser.parse_args()
output = PROJECT / "astranomy/runs" / (args.task + "_high_001")
sys.path[:0] = [str(RUNTIME / "src"), str(RUNTIME)]
os.environ.pop("OPENAI_API_KEY", None)
os.environ.pop("CUDA_VISIBLE_DEVICES", None)
os.environ.update(PYTHONPATH=str(RUNTIME / "src") + ":" + str(RUNTIME),
                  PYTHONDONTWRITEBYTECODE="1", MUJOCO_GL="egl", MUJOCO_EGL_DEVICE_ID=args.gpu,
                  OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")

from astranomy import policy, client, transport, run
from gpt_field import libero
from gpt_field.chatgpt_auth import SubscriptionCredentials

for module in (policy, client, transport, run, libero):
    assert Path(module.__file__).resolve().is_relative_to(RUNTIME), module.__file__
assert policy.MODEL == "gpt-6-astra" and policy.EFFORT == "high"
assert not output.exists(), "A fresh episode output is required"
audit = json.loads((PREPARATION / "source_audit.json").read_text())
for record in audit["runtime_files"]:
    assert hashlib.sha256((RUNTIME / record["path"]).read_bytes()).hexdigest() == record["sha256"]
validation = json.loads((PREPARATION / "validation.json").read_text())
assert validation["passed"] and validation["tasks"][args.task]["reset"]["passed"]
assert validation["tasks"][args.task]["protocol"]["passed"]
print(json.dumps({"event": "isolated_high_runtime_verified", "task_key": args.task,
                  "runtime": str(RUNTIME), "output": str(output),
                  "model": policy.MODEL, "reasoning_effort": policy.EFFORT}), flush=True)
credentials = SubscriptionCredentials(store_dir=PROJECT / ".private/chatgpt_plan")
result = run.execute(output, task_key=args.task, episode_index=0,
                     token_provider=credentials.access_token, video=True)
raise SystemExit(0 if result["evaluation_completed"] else 3)
