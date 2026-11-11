import json, glob, os
for run in sorted(glob.glob('/workspace/gpt_field/runs/agent_speedup_v2_uc71_*')):
    p = run + '/trial/trace.jsonl'
    if not os.path.isfile(p): continue
    prev = None; closes = 0; n = 0
    with open(p) as f:
        for line in f:
            r = json.loads(line)
            if r.get('event') != 'action': continue
            g = r['action'][6] if r.get('action') else None
            if g is None: continue
            if prev is not None and prev < 0 and g > 0:
                closes += 1
            if prev is None and g > 0:
                closes += 1
            prev = g; n += 1
    print(os.path.basename(run)[23:], 'actions', n, 'open->close transitions', closes)
