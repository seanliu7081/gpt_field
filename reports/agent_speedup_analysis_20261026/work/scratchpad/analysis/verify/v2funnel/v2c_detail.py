import json, os, sys
ROOT='/workspace/gpt_field/runs'
RUNS={'t7_001':'agent_speedup_v3_t7_001','g5_002':'agent_speedup_v3_g5_002','base_a3':'v3_completion_baseline_20261026T133633Z/attempts/attempt_003/execution',
      't1_010':'agent_speedup_v3_t1_010','t1_009':'agent_speedup_v3_t1_009','g5_001':'agent_speedup_v3_g5_001','t1_005':'agent_speedup_v3_t1_005','t1_011':'agent_speedup_v3_t1_011','t1_012':'agent_speedup_v3_t1_012'}
def load(k):
    txt=open(os.path.join(ROOT,RUNS[k],'error_traceback.txt')).read()
    key='EvidenceAcquisitionFailed: '
    return json.loads(txt[txt.rfind(key)+len(key):].strip())
for k in sys.argv[1:]:
    d=load(k); b=d['blocker']
    print('=====',k,'status',b.get('status'),'reason',b.get('reason'),'clearance_estimate_m',b.get('clearance_estimate_m'),'interval_padding_m',b.get('interval_padding_m'))
    print('  blocking_reasons',b.get('blocking_reasons'),'execution_status',b.get('execution_status'))
    print('  limiting_pair', json.dumps(b.get('limiting_pair')))
    for w in b.get('coverage_witnesses') or []:
        print('  W', {kk:(round(v,5) if isinstance(v,float) else v) for kk,v in w.items() if kk in ('axis','body_id','body_part','coverage_kind','deficit_m','face','nominal_margin_m','padded_margin_m','uncertainty_m','body_uncertainty_m','uncertainty_margin_m','interpolation_padding_m','region_id','path_fraction','point_outside_eroded_region_union','source')})
    for m in b.get('missing_evidence_requests') or []:
        print('  MER', json.dumps(m)[:300])
    ex=b.get('execution_prefix') or {}
    print('  execution_prefix', {kk:ex.get(kk) for kk in ('status','reason','fraction','supported_fraction')})
    wga=b.get('whole_goal_assessment') or {}
    print('  whole_goal_assessment', {kk:wga.get(kk) for kk in ('status','reason','clearance_estimate_m','allowed_fraction','unknown_fraction')})
    print('  wga.limiting_pair', json.dumps(wga.get('limiting_pair'))[:400])
    for w in wga.get('coverage_witnesses') or []:
        print('  WGA-W', {kk:(round(v,5) if isinstance(v,float) else v) for kk,v in w.items() if kk in ('axis','body_id','body_part','coverage_kind','deficit_m','face','nominal_margin_m','padded_margin_m','uncertainty_m','region_id','path_fraction')})
    cc=b.get('command_candidates') or []
    for c in cc:
        print('  CMD', {kk:(round(v,6) if isinstance(v,float) else v) for kk,v in c.items() if kk in ('clearance_estimate_m','feasible','reason','interval_padding_m','translation_step_m','command_translation_m')})
    sc=b.get('selected_command_candidate')
    if sc: print('  SELECTED', {kk:(round(v,6) if isinstance(v,float) else v) for kk,v in sc.items() if kk in ('clearance_estimate_m','feasible','reason','interval_padding_m','translation_step_m')})
