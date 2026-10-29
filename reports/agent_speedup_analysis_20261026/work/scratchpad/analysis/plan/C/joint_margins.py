"""Plan C: joint-limit margins at every uc7 executive request and the model's stall-cause attribution.
Read-only, stdlib. Joint ranges: /venv/fm_dno/lib/python3.10/site-packages/robosuite/models/assets/robots/panda/robot.xml."""
import json, glob, re
LIM = [(-2.8973, 2.8973), (-1.7628, 1.7628), (-2.8973, 2.8973), (-3.0718, -0.0698),
       (-2.8973, 2.8973), (-0.0175, 3.7525), (-2.8973, 2.8973)]
ROOT = '/workspace/gpt_field/runs/agent_speedup_v2_user_check_libero_10_'
RUNS = ['t0_001','t1_001','t2_002','t3_003','t3_005','t4_001','t5_001','t5_002','t6_009','t6_011','t7_001','t8_001','t9_001']
KW = re.compile(r'elbow|joint|reach|kinematic|singular|straight|extended|stretch', re.I)
rows = []
for key in RUNS:
    reqs = sorted(glob.glob(ROOT + key + '/requests/*.request.json'))
    for rq in reqs:
        d = json.load(open(rq))
        inp = d['payload']['input']
        last = [m for m in inp if m.get('role') == 'user'][-1]
        ctx = None
        for part in last['content']:
            if part.get('type') == 'input_text' and part['text'].startswith('{'):
                try:
                    t = json.loads(part['text'])
                except Exception:
                    continue
                if 'context' in t:
                    ctx = t['context']; break
        if ctx is None:
            continue
        q = ctx['measured_state']['joint_position_rad']
        margins = [min(qi - lo, hi - qi) for qi, (lo, hi) in zip(q, LIM)]
        jmin = min(range(7), key=lambda i: margins[i])
        fb = ctx.get('last_feedback') or {}
        reason = fb.get('reason') or (fb.get('summary') or {}).get('reason')
        # response belief
        resp = rq.replace('.request.json', '.response.json')
        belief = ''
        try:
            r = json.load(open(resp))
            for item in r.get('output', []):
                if item.get('type') == 'message':
                    for c in item.get('content', []):
                        if c.get('type') == 'output_text':
                            belief = json.loads(c['text']).get('belief', '')
        except Exception:
            pass
        rows.append({'run': key, 'req': rq.split('/')[-1][:6], 'step': ctx.get('step'), 'reason': reason,
                     'min_margin_rad': round(margins[jmin], 4), 'joint': jmin + 1,
                     'blames_kinematics': bool(KW.search(belief))})
near = [r for r in rows if r['min_margin_rad'] < 0.05]
stall = [r for r in rows if r['reason'] == 'segment_stall']
print('requests', len(rows), 'with a joint within 0.05 rad of a limit:', len(near))
from collections import Counter
print(' by run/joint', Counter((r['run'], r['joint']) for r in near))
print('stall handoffs', len(stall), ' of which a joint within 0.05 rad:', sum(r['min_margin_rad'] < 0.05 for r in stall))
print(' stall handoffs where model blamed kinematics:', sum(r['blames_kinematics'] for r in stall),
      ' ... and a joint actually within 0.05 rad:', sum(r['blames_kinematics'] and r['min_margin_rad'] < 0.05 for r in stall),
      ' ... and no joint within 0.15 rad:', sum(r['blames_kinematics'] and r['min_margin_rad'] >= 0.15 for r in stall))
for r in stall:
    if r['blames_kinematics'] or r['min_margin_rad'] < 0.05:
        print('  ', r)
json.dump(rows, open('/tmp/claude-0/-workspace/02fb8e60-2ddf-4353-88bc-336305aea415/scratchpad/analysis/plan/C/joint_margins.json', 'w'), indent=1)
