import json, glob, os, math
base = '/workspace/gpt_field/runs'
D = '/tmp/claude-0/-workspace/ba83318d-1178-4524-a098-31607ddcba01/scratchpad/harness_analysis/w1/data/'
runs = {
 't04_t100': 'agent_speedup_v2_uc71_can_obstacle_transit100_t04_20261103_001',
 't08_stoveoff': 'agent_speedup_v2_uc71_stove_off_bottle_obstacle_transit500_cap100_t08_20261103_001',
 'drawer_001': 'agent_speedup_v2_uc71_drawer_bowl_moka_transit500_cap100_20261103_001',
 'drawer_003': 'agent_speedup_v2_uc71_drawer_bowl_moka_transit500_cap100_20261103_003',
}
SC = json.load(open(D + 'stall_class.json'))
def answers(run):
    out = []
    for f in sorted(glob.glob(os.path.join(base, run, 'trial', 'requests', '*.response.json'))):
        d = json.load(open(f)); r = d.get('response', d); txt = None
        for item in r.get('output', []):
            if item.get('type') == 'message':
                for c in item.get('content', []):
                    if c.get('type') == 'output_text': txt = c['text']
        out.append(json.loads(txt))
    return out
for key, run in runs.items():
    A = answers(run)
    for det in SC[key]['details']:
        if det['kind'] not in ('transit_overshoot_false_stall', 'transit_stall_near'): continue
        ci = det['call'] - 1  # index of call triggered by the stall
        prev = A[ci - 1]; nxt = A[ci]
        si = int(det['seg'].split('/')[0]) - 1
        pseg = prev['execution']['segments'][si]
        nsegs = (nxt.get('execution') or {}).get('segments') or []
        first = nsegs[0] if nsegs else None
        same = None
        if first and first['skill'] == 'move_to_pose' and pseg['skill'] == 'move_to_pose':
            same = round(math.dist(first['xyz_m'], pseg['xyz_m']) * 1000, 1)
        print(key, 'call', det['call'], det['kind'], 'seg', det['seg'], 'min/end err', det['min_err_mm'], det['end_err_mm'],
              '| next first seg:', (first['skill'], first['profile']) if first else nxt.get('type'), 'dist to stalled target mm:', same,
              '| next plan:', nxt['plan'][:110])
