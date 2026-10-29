"""Plan C: finger-gap events during close-maintained arm motion in uc7 runs (read-only, stdlib only).
Gap = state[6] - state[7] (m) from episode.events.jsonl 'executed' rows (post-step state).
A 'collapse' = while the last gripper command is close and the arm is in a motion stage, the gap falls
>= 6 mm below its post-close settled value for 3 consecutive samples (mirror of fragments.py:278-289 rise rule)."""
import json, sys
ROOT = '/workspace/gpt_field/runs/agent_speedup_v2_user_check_libero_10_'
RUNS = ['t0_001','t1_001','t2_002','t3_003','t3_005','t4_001','t5_001','t5_002','t6_009','t6_011','t7_001','t8_001','t9_001']
out = {}
for key in RUNS:
    rows = []
    with open(ROOT + key + '/episode.events.jsonl') as f:
        for line in f:
            d = json.loads(line)
            if d.get('event') == 'executed':
                rows.append(d)
    events = []
    cmd = 'open'; close_start = None; settled = None; below = 0; fired = False; floor = None
    for i, d in enumerate(rows):
        a = d.get('action') or []
        st = d.get('state') or []
        if len(a) < 7 or len(st) < 8:
            continue
        g = (st[6] - st[7]) * 1000.0
        new = 'close' if a[6] > 0 else 'open' if a[6] < 0 else cmd
        if new != cmd:
            cmd = new
            close_start = i if cmd == 'close' else None
            settled = None; below = 0; fired = False; floor = None
        if cmd != 'close':
            continue
        n = i - close_start
        if n == 15:  # settled reference after the minimum close time
            settled = g
        if settled is None:
            continue
        # moving? use measured_delta_xyz_m if present
        md = d.get('measured_delta_xyz_m')
        moving = md is not None and sum(x*x for x in md) ** .5 > 2e-4
        if settled - g >= 6.0:
            below += 1
        else:
            below = 0
        if below >= 3 and not fired:
            fired = True
            events.append({'step': d.get('step'), 'settled_gap_mm': round(settled, 1), 'gap_mm': round(g, 1),
                           'moving': moving, 'primitive': d.get('primitive'), 'since_close': n})
    out[key] = events
    print(key, len(rows), 'collapse events:', events)
json.dump(out, open('/tmp/claude-0/-workspace/02fb8e60-2ddf-4353-88bc-336305aea415/scratchpad/analysis/plan/C/gap_events.json', 'w'), indent=1)
