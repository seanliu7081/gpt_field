"""Size of keypose-2 SceneSnapshot.summary() (sent as context.scene) on saved frames; compare with run-001 request sizes."""
import json, sys, glob
import numpy as np
sys.path.insert(0, '/tmp/claude-0/-workspace/ba83318d-1178-4524-a098-31607ddcba01/scratchpad/harness_analysis/w2/scripts')
from k2_common import *  # noqa
from gpt_field.foundation_harness.types import jsonable
rows = []
seen = set()
for rec in records:
    if rec['arrays'] in seen: continue
    seen.add(rec['arrays'])
    scene, _ = restore(rec)
    # restore() rebuilds evidence from the saved record; summary() adds planning_surface_* per object
    s = scene.summary(0.)
    n_scene = len(json.dumps(s, allow_nan=False))
    per_obj = {o['id']: len(json.dumps(o, allow_nan=False)) for o in s['objects']}
    rows.append({'arrays': rec['arrays'], 'objects': len(s['objects']), 'candidates': len(s['candidates']),
                 'scene_summary_chars': n_scene, 'per_object_chars': per_obj})
for r in rows: print(json.dumps(r))
# run-001 (older revision) request fields size
sizes = []
for p in sorted(glob.glob(str(RUN / 'trial/requests/*.request.json'))):
    d = json.load(open(p))
    try:
        content = d['payload']['input'][-1]['content']
        text = next(c['text'] for c in content if c.get('type') == 'input_text')
        sizes.append(len(text))
    except Exception as e:
        sizes.append(None)
print('run001 last-user-turn text chars per request', sizes)
(W2 / 'k2_summary_size.json').write_text(json.dumps({'keypose2_scene_summary': rows, 'run001_user_text_chars': sizes}, indent=1))
