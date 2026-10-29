"""Plan C: execute decisions whose planned set_gripper stage was dropped because an earlier arm stage
ended the fragment (stall / budget / envelope). Read-only, stdlib. Uses each run's request files: the
request after decision i carries last_feedback (possibly compacted: summary + stage_outcomes)."""
import json, glob, re
ROOT = '/workspace/gpt_field/runs/agent_speedup_v2_user_check_libero_10_'
RUNS = ['t0_001','t1_001','t2_002','t3_003','t3_005','t4_001','t5_001','t5_002','t6_009','t6_011','t7_001','t8_001','t9_001']
def ctx_of(path):
    d = json.load(open(path))
    last = [m for m in d['payload']['input'] if m.get('role') == 'user'][-1]
    for part in last['content']:
        if part.get('type') == 'input_text' and part['text'].startswith('{'):
            try:
                t = json.loads(part['text'])
            except Exception:
                continue
            if 'context' in t:
                return t['context']
def resp_of(path):
    r = json.load(open(path.replace('.request.json', '.response.json')))
    for item in r.get('output', []):
        if item.get('type') == 'message':
            for c in item.get('content', []):
                if c.get('type') == 'output_text':
                    return json.loads(c['text'])
tot = {'execute_with_gripper_stage': 0, 'dropped': 0, 'dropped_after_stall': 0, 'runs': {}}
for key in RUNS:
    reqs = sorted(glob.glob(ROOT + key + '/requests/*.request.json'))
    n = 0; dropped = 0; stall = 0; detail = []
    for i, rq in enumerate(reqs[:-1]):
        a = resp_of(rq)
        if not a or a.get('type') != 'execute':
            continue
        segs = a['execution']['segments']
        gi = [k for k, s in enumerate(segs) if s['skill'] == 'set_gripper']
        if not gi:
            continue
        n += 1
        fb = ctx_of(reqs[i + 1]).get('last_feedback') or {}
        summ = fb.get('summary') or fb
        reason = summ.get('reason')
        ev = fb.get('evidence') or summ.get('evidence') or {}
        done = ev.get('completed_segment_indices')
        if done is None:
            outs = fb.get('stage_outcomes') or []
            done = sorted({int(m.group(1)) for o in outs if o.get('status') == 'succeeded'
                           for m in [re.search(r'segments\[(\d+)\]', o.get('path', ''))] if m})
        first_g = gi[0]
        if first_g not in done and reason != 'fragment_complete':
            dropped += 1; stall += reason == 'segment_stall'
            detail.append((a.get('type'), reason, [s['skill'] for s in segs], done))
    tot['execute_with_gripper_stage'] += n; tot['dropped'] += dropped; tot['dropped_after_stall'] += stall
    tot['runs'][key] = {'with_gripper': n, 'dropped': dropped, 'after_stall': stall, 'detail': detail}
    print(key, 'execute-with-gripper', n, 'gripper stage dropped', dropped, '(after stall', stall, ')')
print({k: v for k, v in tot.items() if k != 'runs'})
json.dump(tot, open('/tmp/claude-0/-workspace/02fb8e60-2ddf-4353-88bc-336305aea415/scratchpad/analysis/plan/C/dropped_gripper.json', 'w'), indent=1)
