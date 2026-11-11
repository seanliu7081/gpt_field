import json, sys, os, statistics as st
base = '/workspace/gpt_field/runs'
runs = {
 't04_t100': 'agent_speedup_v2_uc71_can_obstacle_transit100_t04_20261103_001',
 't08_t100': 'agent_speedup_v2_uc71_bottle_obstacle_transit100_t08_20261103_001',
 't08_stoveoff': 'agent_speedup_v2_uc71_stove_off_bottle_obstacle_transit500_cap100_t08_20261103_001',
 'drawer_001': 'agent_speedup_v2_uc71_drawer_bowl_moka_transit500_cap100_20261103_001',
 'drawer_003': 'agent_speedup_v2_uc71_drawer_bowl_moka_transit500_cap100_20261103_003',
}
res = {}
for key, run in runs.items():
    ev = os.path.join(base, run, 'trial', 'episode.events.jsonl')
    dec = {}; exe = {}
    with open(ev) as f:
        for line in f:
            d = json.loads(line)
            e = d.get('event')
            if e in ('agent_action', 'skill_decision'):
                dec[d['step']] = (d['wall_elapsed_s'], bool(d.get('api_calls_in_act')))
            elif e == 'executed':
                exe[d['step']] = d['wall_elapsed_s']
    steps = sorted(exe)
    gaps = []
    for s in steps[1:]:
        if s in dec and not dec[s][1] and (s-1) in exe:
            gaps.append((s, exe[s] - exe[s-1]))
    n = len(gaps)
    q = max(1, n // 5)
    quint = [st.mean(g for _, g in gaps[i*q:(i+1)*q]) for i in range(5)]
    res[key] = dict(nonapi_steps=n, mean=st.mean(g for _, g in gaps), quintiles=[round(x, 4) for x in quint])
    print(key, 'non-API step wall gap (incl. video): n=%d mean %.4f s; by episode quintile: %s' % (n, res[key]['mean'], res[key]['quintiles']))
json.dump(res, open('/tmp/claude-0/-workspace/ba83318d-1178-4524-a098-31607ddcba01/scratchpad/harness_analysis/w1/data/stepgrowth.json', 'w'), indent=1)
