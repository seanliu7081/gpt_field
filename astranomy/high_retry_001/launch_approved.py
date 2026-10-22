"""One user-authorized high-effort retry using a frozen Astranomy runtime."""
from pathlib import Path
import hashlib
import json
import os
import sys

PROJECT = Path('/workspace/gpt_field')
PREPARATION = PROJECT / 'astranomy/high_retry_001'
RUNTIME = PREPARATION / 'runtime'
OUTPUT = PROJECT / 'astranomy/runs/drawer_bowl_high_001'
sys.path[:0] = [str(RUNTIME / 'src'), str(RUNTIME)]
os.environ.pop('OPENAI_API_KEY', None)
os.environ.pop('CUDA_VISIBLE_DEVICES', None)
os.environ.update(PYTHONPATH=str(RUNTIME / 'src') + ':' + str(RUNTIME),
                  PYTHONDONTWRITEBYTECODE='1', MUJOCO_GL='egl', MUJOCO_EGL_DEVICE_ID='2',
                  OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')

from astranomy import policy, client, transport, run
from gpt_field import libero
from gpt_field.chatgpt_auth import SubscriptionCredentials

for module in (policy, client, transport, run, libero):
    assert Path(module.__file__).resolve().is_relative_to(RUNTIME), module.__file__
assert policy.MODEL == 'gpt-6-astra' and policy.EFFORT == 'high'
assert not OUTPUT.exists(), 'A fresh episode output is required'
audit = json.loads((PREPARATION / 'source_audit.json').read_text())
for record in audit['records']:
    assert hashlib.sha256((RUNTIME / record['path']).read_bytes()).hexdigest() == record['high_sha256']
assert json.loads((PREPARATION / 'validation.json').read_text())['offline_protocol']['passed']
print(json.dumps({'event':'isolated_high_runtime_verified','runtime':str(RUNTIME),
                  'output':str(OUTPUT),'model':policy.MODEL,'reasoning_effort':policy.EFFORT}), flush=True)
credentials = SubscriptionCredentials(store_dir=PROJECT / '.private/chatgpt_plan')
result = run.execute(OUTPUT, task_key='drawer_bowl', episode_index=0,
                     token_provider=credentials.access_token, video=True)
raise SystemExit(0 if result['evaluation_completed'] else 3)
