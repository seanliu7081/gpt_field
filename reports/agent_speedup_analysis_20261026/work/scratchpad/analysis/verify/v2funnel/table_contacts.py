import json, os
ROOT='/workspace/gpt_field/runs'
RUNS = {f't1_{i:03d}':f'agent_speedup_v3_t1_{i:03d}' for i in range(1,13)}
RUNS['base_a3']='v3_completion_baseline_20261026T133633Z/attempts/attempt_003/execution'
RUNS['base_a1']='v3_completion_baseline_20261026T133633Z/attempts/attempt_001/execution'
for k,r in RUNS.items():
    p=os.path.join(ROOT,r,'collision.json')
    if not os.path.exists(p): print(k,'no collision.json'); continue
    d=json.load(open(p))
    g=d.get('generic') or {}
    fb=g.get('first_by_class') or {}
    rows=[]
    for cls,e in fb.items():
        names=[e.get('geom1',{}).get('body'), e.get('geom2',{}).get('body')]
        if any(n and 'table' in n for n in names):
            rows.append(f"{cls}@a{e.get('action_index')} z={e['contact_pos'][2]:.5f} other={[n for n in names if 'table' not in (n or '')]}")
    tbl_pairs=[(p['class'],p['first_action_index'],p['pair']) for p in sum((g.get('pairs_by_class') or {}).values(),[]) if any('geom_10' in x or 'table' in x for x in p['pair'])]
    print(k, '| first table-involving contacts:', rows)
    print('    table pairs:', [(c,a,[x for x in pr if 'geom_10' not in x]) for c,a,pr in tbl_pairs][:8])
