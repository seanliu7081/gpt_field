import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from table_track import top_z
ROOT='/workspace/gpt_field/runs'
RUNS = {f't1_{i:03d}':f'agent_speedup_v3_t1_{i:03d}' for i in range(3,13)}
RUNS['base_a3']='v3_completion_baseline_20261026T133633Z/attempts/attempt_003/execution'
for k,r in RUNS.items():
    st=json.load(open(os.path.join(ROOT,r,'agent_stats.json')))
    sg=st.get('shared_geometry') or {}
    ents=sg.get('entities') or {}
    for ident,e in ents.items():
        if e.get('role')=='support_surface':
            tz=top_z(e) if e.get('boxes') else None
            extra={kk:e.get(kk) for kk in ('source','observation_step','stale','stale_reason','local_pose_evidence','local_pose_revision') if kk in e}
            print(k, ident, 'top_z=',None if tz is None else round(tz,4), 'unc=',e.get('uncertainty_m'), json.dumps(extra)[:300])
