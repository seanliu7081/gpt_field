import json, sys, os
sys.path.insert(0, os.path.dirname(__file__))
ROOT='/workspace/gpt_field/runs'
from parse_eaf import RUNS, load_eaf
order=['t1_005','t1_009','t1_010','t7_001','g5_001','g5_002','t1_011','t1_012','base_a3']
for k in order:
    p=os.path.join(ROOT,RUNS[k],'error_traceback.txt')
    d,_=load_eaf(p)
    b=d['blocker']
    st=json.load(open(os.path.join(ROOT,RUNS[k],'agent_stats.json')))
    ep=st.get('evidence_pause') or {}
    same = (json.dumps(ep,sort_keys=True)==json.dumps(d,sort_keys=True))
    print(f"== {k}: step={d.get('step')} reason={d['reason'][:110]!r}")
    print(f"   agent_stats.evidence_pause identical to traceback JSON: {same}; ep.reason={ep.get('reason','')[:60]!r}")
    print(f"   blocker.status={b.get('status')} reason={b.get('reason')} clearance_estimate_m={b.get('clearance_estimate_m')} allowed_fraction={b.get('allowed_fraction')} unknown_fraction={b.get('unknown_fraction')}")
    print(f"   execution_status={b.get('execution_status')} execution_allowed={b.get('execution_allowed')} blocking_reasons={b.get('blocking_reasons')}")
    ex=b.get('execution_prefix') or {}
    print(f"   execution_prefix: status={ex.get('status')} reason={ex.get('reason')} fraction={ex.get('fraction')}")
    print(f"   n_coverage_witnesses={len(b.get('coverage_witnesses') or [])} n_cmd_candidates={len(b.get('command_candidates') or []) if b.get('command_candidates') is not None else None} missing_evidence_requests={len(b.get('missing_evidence_requests') or [])}")
    lp=b.get('limiting_pair')
    print(f"   limiting_pair={json.dumps(lp)[:400]}")
